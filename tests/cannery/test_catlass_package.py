import builtins
import math
import subprocess
import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from test_package import loader, package


def catlass_metadata(metadata):
    metadata.update(
        schema_version=2,
        abi="catlass-unary-f32-v1",
        specializations=[
            {"shape": [17], "dtype": "float32"},
            {"shape": [3, 7], "dtype": "float32"},
            {"shape": [0], "dtype": "float32"},
        ],
    )


class Tensor:
    def __init__(self, shape=(17,), dtype="float32", device=0, pointer=100):
        self.shape, self.dtype, self.pointer = shape, dtype, pointer
        self.device = SimpleNamespace(type="npu", index=device)
        self.recorded = []
        self.materialized = None

    def contiguous(self):
        self.materialized = Tensor(
            self.shape, self.dtype, self.device.index, self.pointer + 1
        )
        return self.materialized

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return math.prod(self.shape)

    def record_stream(self, stream):
        self.recorded.append(stream)


def torch_stub():
    stream = SimpleNamespace(npu_stream=99)
    npu = SimpleNamespace(
        current_device=lambda: 0,
        device=lambda _: nullcontext(),
        current_stream=lambda _: stream,
    )
    return SimpleNamespace(
        Tensor=Tensor,
        float32="float32",
        npu=npu,
        empty_like=lambda x: Tensor(x.shape, x.dtype, x.device.index, 200),
    )


class CatlassPackageTests(unittest.TestCase):
    def test_schema_versions_and_shape_budgets_rejected_before_loading(self):
        changes = [
            lambda m: m.update(schema_version=1),
            lambda m: m.update(abi="unary-f32-v1"),
            lambda m: m.update(specializations=[]),
            lambda m: m.update(
                specializations=[{"shape": [1], "dtype": "float32"}] * 5
            ),
            lambda m: m.update(
                specializations=[{"shape": [17], "dtype": "float32"}] * 2
            ),
        ]
        for shape, dtype in [
            ([], "float32"),
            ([True], "float32"),
            ([1.5], "float32"),
            ([-1], "float32"),
            ([1024, 1025], "float32"),
            ([1] * 5, "float32"),
            ([17], "float16"),
            ([0], "float32"),
        ]:
            changes.append(
                lambda m, s=shape, d=dtype: m.update(
                    specializations=[{"shape": s, "dtype": d}]
                )
            )
        for change in changes:

            def alter(m, mutate=change):
                catlass_metadata(m)
                mutate(m)

            data, sha, _ = package(alter)
            with (
                self.subTest(change=change),
                patch.object(
                    loader.ctypes, "CDLL", side_effect=AssertionError("no load")
                ),
                self.assertRaises(loader.CanneryPackageError),
            ):
                loader.verify_cannery_package(data, sha)

    def run_kernel(self, exercise, status=0, catlass=True):
        data, sha, metadata = package(catlass_metadata if catlass else None)
        launch = MagicMock(return_value=status)
        lib = SimpleNamespace(cannery_launch=launch)
        torch = torch_stub()
        original_import = builtins.__import__

        def offline_import(name, *args, **kwargs):
            if name.startswith("catlass"):
                raise AssertionError("CATLASS/JIT must not be imported")
            return original_import(name, *args, **kwargs)

        with (
            patch.dict(sys.modules, {"torch": torch, "torch_npu": SimpleNamespace()}),
            patch.object(
                loader, "runtime_fingerprint", return_value=metadata["runtime"]
            ),
            patch.object(loader.ctypes, "CDLL", return_value=lib),
            patch.object(builtins, "__import__", side_effect=offline_import),
            patch.object(subprocess, "Popen", side_effect=AssertionError("offline")),
        ):
            kernel = loader.load_cannery_package(data, sha, trust_remote_code=True)
            try:
                exercise(kernel, launch, torch)
            finally:
                kernel._directory.cleanup()

    def test_declared_noncontiguous_input_preserves_shape_dtype_and_device(self):
        def exercise(kernel, launch, torch):
            x = Tensor((3, 7))
            y = kernel.forward(x)
            self.assertEqual((y.shape, y.dtype, y.device.index), ((3, 7), "float32", 0))
            launch.assert_called_once_with(99, 101, 200, 21)
            self.assertEqual(len(x.materialized.recorded), 1)
            self.assertEqual(len(y.recorded), 1)
            self.assertIs(launch.restype, loader.ctypes.c_int)

        self.run_kernel(exercise)

    def test_empty_variant_never_calls_native_and_domain_is_frozen(self):
        def exercise(kernel, launch, torch):
            kernel.metadata["specializations"].append(
                {"shape": [18], "dtype": "float32"}
            )
            self.assertEqual(kernel.forward(Tensor((0,))).shape, (0,))
            for tensor in [
                Tensor((18,)),
                Tensor((17,), device=1),
                Tensor(dtype="float16"),
            ]:
                with self.assertRaises(loader.CanneryPackageError):
                    kernel.forward(tensor)
            launch.assert_not_called()

        self.run_kernel(exercise)

    def test_native_error_propagates_after_recording_inflight_storage(self):
        def exercise(kernel, launch, torch):
            output = Tensor(pointer=200)
            torch.empty_like = lambda x: output
            x = Tensor()
            with self.assertRaisesRegex(loader.CanneryPackageError, "507033"):
                kernel.forward(x)
            self.assertEqual(len(x.materialized.recorded), 1)
            self.assertEqual(len(output.recorded), 1)

        self.run_kernel(exercise, status=507033)

    def test_m3_void_return_and_unrestricted_shape_remain_compatible(self):
        def exercise(kernel, launch, torch):
            self.assertEqual(kernel.forward(Tensor((18,))).shape, (18,))
            self.assertIsNone(launch.restype)

        self.run_kernel(exercise, status=None, catlass=False)
