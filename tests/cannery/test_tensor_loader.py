import math
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from test_tensor_signatures import variant

from cannery_kernels import CanneryPackageError
from cannery_kernels.tensor_loader import TensorKernel


class FakeTensor:
    next_pointer = 4096

    def __init__(self, shape, dtype="float16", device=None):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = device or SimpleNamespace(type="npu", index=0)
        self.pointer = FakeTensor.next_pointer
        FakeTensor.next_pointer += 4096
        self.contiguous, self.streams = True, []

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return 2 if self.dtype == "float16" else 4

    def record_stream(self, stream):
        self.streams.append(stream)


class TensorLoaderTests(unittest.TestCase):
    def setUp(self):
        self.stream = SimpleNamespace(npu_stream=99)
        torch = SimpleNamespace(
            Tensor=FakeTensor,
            float16="float16",
            int32="int32",
            empty=lambda shape, dtype, device: FakeTensor(shape, dtype, device),
            npu=SimpleNamespace(
                current_device=lambda: 0,
                device=lambda _: nullcontext(),
                current_stream=lambda _: self.stream,
            ),
        )
        self.native = MagicMock(return_value=0)
        modules = patch.dict(
            "sys.modules", {"torch": torch, "torch_npu": SimpleNamespace()}
        )
        modules.start()
        library = patch(
            "cannery_kernels.tensor_loader.ctypes.CDLL",
            return_value=SimpleNamespace(
                cannery_launch_tensor_v2=self.native,
            ),
        )
        library.start()
        self.addCleanup(modules.stop)
        self.addCleanup(library.stop)
        self.kernel = TensorKernel(b"fake-never-executed", {"variants": [variant()]})
        self.addCleanup(self.kernel._directory.cleanup)
        self.x, self.cache = FakeTensor((2, 3)), FakeTensor((4, 3))

    def test_named_alias_outputs_and_native_descriptor_order(self):
        outputs = self.kernel.forward(self.x, self.cache, offset=1)
        self.assertIs(outputs["cache_out"], self.cache)
        self.assertEqual(
            (outputs["indices"].dtype, outputs["indices"].shape), ("int32", (2,))
        )
        stream, ni, inputs, no, native_outputs, ns, scalars = self.native.call_args.args
        self.assertEqual((stream, ni, no, ns), (99, 2, 2, 1))
        self.assertEqual(inputs[0].data, self.x.pointer)
        self.assertEqual(native_outputs[0].data, self.cache.pointer)
        self.assertEqual(scalars[0].bits, 1)
        for tensor in [self.x, self.cache, outputs["indices"]]:
            self.assertIn(self.stream, tensor.streams)

    def test_invalid_interface_domains_strides_devices_and_overlap_reject_before_native(
        self,
    ):
        for operation in [
            lambda: self.kernel.forward(self.x, self.cache),
            lambda: self.kernel.forward(self.x, self.cache, offset=True),
            lambda: self.kernel.forward(self.x, self.cache, offset=2),
            lambda: self.kernel.forward(self.x, self.cache, offset=1, extra=0),
            lambda: self.kernel.forward(FakeTensor((3, 2)), self.cache, offset=1),
            lambda: self.kernel.forward(
                FakeTensor((2, 3), "int32"), self.cache, offset=1
            ),
            lambda: self.kernel.forward("x", self.cache, offset=1),
        ]:
            with self.assertRaises(CanneryPackageError):
                operation()
        self.x.contiguous = False
        with self.assertRaises(CanneryPackageError):
            self.kernel.forward(self.x, self.cache, offset=1)
        self.x.contiguous = True
        self.cache.device.index = 1
        with self.assertRaises(CanneryPackageError):
            self.kernel.forward(self.x, self.cache, offset=1)
        self.cache.device.index = 0
        self.cache.pointer = self.x.pointer + 2
        with self.assertRaisesRegex(CanneryPackageError, "overlaps"):
            self.kernel.forward(self.x, self.cache, offset=1)
        self.native.assert_not_called()

    def test_partial_enqueue_failure_records_streams_and_allows_later_call(self):
        self.native.return_value = 507000
        with self.assertRaisesRegex(CanneryPackageError, "507000"):
            self.kernel.forward(self.x, self.cache, offset=1)
        self.assertIn(self.stream, self.x.streams)
        self.assertIn(self.stream, self.cache.streams)
        self.native.return_value = 0
        self.kernel.forward(self.x, self.cache, offset=1)
        self.assertEqual(self.native.call_count, 2)

    def test_native_exception_records_inflight_storage(self):
        self.native.side_effect = RuntimeError("device disconnected")
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            self.kernel.forward(self.x, self.cache, offset=1)
        self.assertIn(self.stream, self.x.streams)

    def test_missing_bridge_fails_cleanly(self):
        with patch(
            "cannery_kernels.tensor_loader.ctypes.CDLL", return_value=SimpleNamespace()
        ):
            with self.assertRaisesRegex(
                CanneryPackageError, "missing its native bridge"
            ):
                TensorKernel(b"fake", {"variants": [variant()]})
