import hashlib
import importlib.util
import io
import json
import tarfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

MODULE = Path(__file__).resolve().parents[2] / "cannery-runtime/src/cannery_kernels/__init__.py"
spec = importlib.util.spec_from_file_location("cannery_loader", MODULE)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


def package(change=None, extra=False):
    library = b"fixture, never executed"
    metadata = {
        "schema_version": 1,
        "abi": "unary-f32-v1",
        "device": "a5",
        "source_sha256": "a" * 64,
        "library_sha256": hashlib.sha256(library).hexdigest(),
        "compiler": "test-bisheng",
        "runtime": {
            "machine": "aarch64",
            "python": "3.12",
            "torch": "2.9.0+cpu",
            "torch_npu": "2.9.0.post9",
            "cann": "9.3.0",
            "soc": "Ascend950",
        },
    }
    if change:
        change(metadata)
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w") as archive:
        for name, value in [
            ("metadata.json", json.dumps(metadata).encode()),
            ("kernel.so", library),
        ]:
            entry = tarfile.TarInfo(name)
            entry.size = len(value)
            archive.addfile(entry, io.BytesIO(value))
        if extra:
            archive.addfile(tarfile.TarInfo("../escape"))
    data = result.getvalue()
    return data, hashlib.sha256(data).hexdigest(), metadata


class PackageTests(unittest.TestCase):
    def test_a3_native_package_and_cross_device_rejection_before_loading(self):
        def a3(metadata):
            metadata["device"] = "a3"
            metadata["runtime"].update(soc="Ascend910B4-1", python="3.11", cann="9.1.0")

        data, sha, metadata = package(a3)
        self.assertEqual(loader.verify_cannery_package(data, sha)[0], metadata)
        with patch.object(loader, "runtime_fingerprint", return_value=metadata["runtime"]):
            library = SimpleNamespace(cannery_launch=MagicMock())
            with patch.object(loader.ctypes, "CDLL", return_value=library):
                kernel = loader.load_cannery_package(data, sha, trust_remote_code=True)
                kernel._directory.cleanup()
        for device, soc in [
            ("a3", "Ascend950"),
            ("a5", "Ascend910B4-1"),
            ("a3", "unknown"),
        ]:
            bad, digest, _ = package(
                lambda m, device=device, soc=soc: (
                    m.update(device=device),
                    m["runtime"].update(soc=soc),
                )
            )
            with self.assertRaises(loader.CanneryPackageError):
                loader.verify_cannery_package(bad, digest)
        with patch.object(loader.ctypes, "CDLL", side_effect=AssertionError("must not load")):
            for change in [{"soc": "Ascend950"}, {"cann": "9.2.0"}, {"python": "3.12"}]:
                with patch.object(
                    loader,
                    "runtime_fingerprint",
                    return_value=metadata["runtime"] | change,
                ):
                    with self.assertRaises(loader.CanneryPackageError):
                        loader.load_cannery_package(data, sha, trust_remote_code=True)

    def test_verified_and_no_execution_before_trust_or_compatibility(self):
        data, sha, metadata = package()
        self.assertEqual(loader.verify_cannery_package(data, sha)[0], metadata)
        with patch.object(loader.ctypes, "CDLL", side_effect=AssertionError("must not load")):
            with self.assertRaisesRegex(loader.CanneryPackageError, "trust_remote_code"):
                loader.load_cannery_package(data, sha)
            with patch.object(
                loader,
                "runtime_fingerprint",
                return_value={**metadata["runtime"], "machine": "x86_64"},
            ):
                with self.assertRaises(loader.CanneryPackageError):
                    loader.load_cannery_package(data, sha, trust_remote_code=True)
        with patch.object(loader, "runtime_fingerprint", return_value=metadata["runtime"]):
            library = SimpleNamespace(cannery_launch=MagicMock())
            with patch.object(loader.ctypes, "CDLL", return_value=library) as native:
                with patch(
                    "socket.create_connection",
                    side_effect=AssertionError("network forbidden"),
                ):
                    kernel = loader.load_cannery_package(data, sha, trust_remote_code=True)
                native.assert_called_once()
                self.assertEqual(kernel.metadata, metadata)
                kernel._directory.cleanup()

    def test_corruption_unsupported_abi_extra_member_and_missing_runtime(self):
        for change in [
            lambda m: m.update(abi="unknown"),
            lambda m: m.update(schema_version=True),
            lambda m: m.update(library_sha256="b" * 64),
            lambda m: m.update(source_sha256="z" * 64),
            lambda m: m.update(source_sha256=["a"] * 64),
            lambda m: m.update(runtime={}),
        ]:
            data, sha, _ = package(change)
            with self.assertRaises(loader.CanneryPackageError):
                loader.verify_cannery_package(data, sha)
        data, sha, _ = package(extra=True)
        with self.assertRaises(loader.CanneryPackageError):
            loader.verify_cannery_package(data, sha)
        data, sha, _ = package()
        with self.assertRaises(loader.CanneryPackageError):
            loader.verify_cannery_package(data + b"corrupt", sha)
        with patch.object(loader, "runtime_fingerprint", side_effect=ImportError):
            with self.assertRaisesRegex(loader.CanneryPackageError, "runtime unavailable"):
                loader.load_cannery_package(data, sha, trust_remote_code=True)

    def test_native_load_failure_cleans_scratch(self):
        _, _, metadata = package()
        with patch.object(loader.ctypes, "CDLL", side_effect=OSError("bad elf")):
            with self.assertRaisesRegex(loader.CanneryPackageError, "entry point unavailable"):
                loader.CanneryKernel(b"bad elf", metadata)


if __name__ == "__main__":
    unittest.main()
