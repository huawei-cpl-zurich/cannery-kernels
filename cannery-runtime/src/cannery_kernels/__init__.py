"""Offline loading of pinned Cannery A5 packages inside an isolated runtime.

This entry point never resolves Hub repositories or downloads dependencies.
Native libraries execute code: callers must explicitly trust the package and
run untrusted submissions in a device-restricted job container.
"""

import ctypes
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

MAX_PACKAGE_BYTES = 64 * 1024 * 1024


class CanneryPackageError(ValueError):
    """A package is corrupt, unsupported, or incompatible with this runtime."""


def runtime_fingerprint() -> dict[str, str]:
    from torch_npu.utils.collect_env import get_cann_version

    return {
        "machine": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": importlib.metadata.version("torch"),
        "torch_npu": importlib.metadata.version("torch-npu"),
        "cann": str(get_cann_version()),
        "soc": __import__("torch").npu.get_device_name(),
    }


def verify_cannery_package(package: bytes, sha256: str) -> tuple[dict[str, Any], bytes]:
    """Verify bounded, uncompressed tar contents without extracting author paths."""
    if (
        len(package) > MAX_PACKAGE_BYTES
        or len(sha256) != 64
        or hashlib.sha256(package).hexdigest() != sha256
    ):
        raise CanneryPackageError("Package digest or size mismatch")
    try:
        with tarfile.open(fileobj=io.BytesIO(package), mode="r:") as archive:
            contents = {}
            for member in archive:
                if member.name not in {"metadata.json", "kernel.so"} or member.name in contents:
                    raise ValueError("Unexpected package members")
                limit = 16384 if member.name == "metadata.json" else MAX_PACKAGE_BYTES - 16384
                if not member.isfile() or member.size <= 0 or member.size > limit:
                    raise ValueError("Invalid package member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Missing package member")
                contents[member.name] = stream.read(limit + 1)
                if len(contents[member.name]) != member.size:
                    raise ValueError("Truncated package member")
        if set(contents) != {"metadata.json", "kernel.so"}:
            raise ValueError("Missing package members")
        metadata = json.loads(contents["metadata.json"])
        if (
            set(metadata)
            != {
                "schema_version",
                "abi",
                "device",
                "source_sha256",
                "library_sha256",
                "runtime",
                "compiler",
            }
            or type(metadata["schema_version"]) is not int
            or metadata["schema_version"] != 1
            or metadata["abi"] != "unary-f32-v1"
            or metadata["device"] != "a5"
            or not isinstance(metadata["compiler"], str)
            or not metadata["compiler"]
            or len(metadata["source_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in metadata["source_sha256"])
            or set(metadata["runtime"])
            != {"machine", "python", "torch", "torch_npu", "cann", "soc"}
            or not all(isinstance(v, str) and v for v in metadata["runtime"].values())
            or hashlib.sha256(contents["kernel.so"]).hexdigest() != metadata["library_sha256"]
        ):
            raise ValueError("Invalid package metadata")
        return metadata, contents["kernel.so"]
    except (
        tarfile.TarError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,
        RecursionError,
    ):
        raise CanneryPackageError("Invalid Cannery binary package") from None


class CanneryKernel:
    """Loaded unary-f32-v1 implementation; keep alive while calls are in flight."""

    def __init__(self, library: bytes, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self._directory = tempfile.TemporaryDirectory(prefix="cannery-kernel-")
        path = Path(self._directory.name) / "kernel.so"
        try:
            path.write_bytes(library)
            path.chmod(0o500)
            self._library = ctypes.CDLL(str(path))
            self._launch = self._library.cannery_launch
            self._launch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint64,
            ]
            self._launch.restype = None
        except (OSError, AttributeError):
            self._directory.cleanup()
            raise CanneryPackageError(
                "Native library or unary-f32-v1 entry point unavailable"
            ) from None

    def forward(self, x: Any) -> Any:
        import torch
        import torch_npu  # noqa: F401 - registers the NPU backend

        if not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or x.device.type != "npu":
            raise CanneryPackageError("unary-f32-v1 expects a float32 NPU tensor")
        with torch.npu.device(x.device):
            contiguous = x.contiguous()
            output = torch.empty_like(contiguous)
            if x.numel():
                stream = torch.npu.current_stream(x.device)
                self._launch(stream.npu_stream, contiguous.data_ptr(), output.data_ptr(), x.numel())
                contiguous.record_stream(stream)
                output.record_stream(stream)
            return output


def load_cannery_package(
    package: bytes, sha256: str, *, trust_remote_code: bool = False
) -> CanneryKernel:
    """Load verified local bytes; no network, dependency resolution, or compilation."""
    if not trust_remote_code:
        raise CanneryPackageError("Native package loading requires trust_remote_code=True")
    metadata, library = verify_cannery_package(package, sha256)
    try:
        current = runtime_fingerprint()
    except (ImportError, importlib.metadata.PackageNotFoundError):
        raise CanneryPackageError("Required CANN/PyTorch NPU runtime unavailable") from None
    if (
        current["machine"] != "aarch64"
        or not current["soc"].startswith("Ascend950")
        or metadata["runtime"] != current
    ):
        raise CanneryPackageError("Package runtime fingerprint mismatch")
    return CanneryKernel(library, metadata)
