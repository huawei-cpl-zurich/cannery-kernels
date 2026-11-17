"""Offline tensor-v2 loading; no source/JIT execution or dependency resolution."""

import ctypes
import tempfile
from pathlib import Path
from typing import Any

from . import CanneryPackageError
from .tensor_abi import (
    ScalarDescriptor,
    TensorDescriptor,
    scalar_descriptor,
    tensor_descriptor,
    validate_argument_budget,
)
from .tensor_signatures import parse_tensor_variants


class TensorKernel:
    """Static named tensor ABI. Retain this object until its stream completes."""

    def __init__(self, library: bytes, metadata: dict[str, Any]) -> None:
        import torch
        import torch_npu  # noqa: F401 - registers NPU

        self.metadata = metadata
        self._variants = parse_tensor_variants(metadata["variants"])
        self._device = torch.npu.current_device()
        self._directory = tempfile.TemporaryDirectory(prefix="cannery-tensor-")
        path = Path(self._directory.name) / "kernel.so"
        try:
            path.write_bytes(library)
            path.chmod(0o500)
            self._library = ctypes.CDLL(str(path))
            self._launch = self._library.cannery_launch_tensor_v2
            self._launch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(TensorDescriptor),
                ctypes.c_uint32,
                ctypes.POINTER(TensorDescriptor),
                ctypes.c_uint32,
                ctypes.POINTER(ScalarDescriptor),
            ]
            self._launch.restype = ctypes.c_int32
        except (OSError, AttributeError):
            self._directory.cleanup()
            raise CanneryPackageError("Tensor library is missing its native bridge") from None
        try:
            prepare = getattr(self._library, "cannery_prepare_tensor_v2", None)
            if prepare is not None:
                prepare.argtypes = []
                prepare.restype = ctypes.c_int32
                with torch.npu.device(self._device):
                    status = prepare()
                if status != 0:
                    raise CanneryPackageError(f"Tensor native preparation failed: {status}")
        except Exception:
            self._directory.cleanup()
            raise

    def forward(self, *inputs: Any, **scalars: Any) -> dict[str, Any]:
        """Return named outputs, including declared mutated input aliases.

        Tensors are positional in the declared order; scalars are keyword-only.
        Only admitted contiguous NPU tensors are accepted. Inputs are read-only
        unless an output explicitly aliases them. No implicit copies or casts.
        """
        import torch
        import torch_npu

        interface = self._variants[0]
        if len(inputs) != len(interface.inputs) or set(scalars) != {
            s.name for s in interface.scalars
        }:
            raise CanneryPackageError("Tensor/scalar arguments do not match the named interface")
        for tensor in inputs:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "npu"
                or tensor.device.index != self._device
                or not tensor.is_contiguous()
                or torch_npu.get_npu_format(tensor) != 2
            ):
                raise CanneryPackageError(
                    "Expected contiguous ND-format NPU tensors on the bound device"
                )
            if tensor.is_neg() or tensor.is_conj():
                raise CanneryPackageError(
                    "Native ABI rejects unresolved negative or conjugate views"
                )
        try:
            scalar_bits = tuple(
                bytes(scalar_descriptor(s.kind, scalars[s.name])) for s in interface.scalars
            )
        except ValueError:
            raise CanneryPackageError("Scalar argument has an invalid type or value") from None
        selected = next(
            (
                v
                for v in self._variants
                if all(
                    tuple(t.shape) == spec.shape and t.dtype == getattr(torch, spec.dtype)
                    for t, spec in zip(inputs, v.inputs, strict=True)
                )
                and scalar_bits
                == tuple(bytes(scalar_descriptor(s.kind, s.value)) for s in v.scalars)
            ),
            None,
        )
        if selected is None:
            raise CanneryPackageError("Arguments are outside the admitted static variants")
        aliases = {out.alias for out in selected.outputs if out.alias is not None}
        for i, (left, left_spec) in enumerate(zip(inputs, selected.inputs, strict=True)):
            for right, right_spec in zip(inputs[i + 1 :], selected.inputs[i + 1 :], strict=True):
                if left_spec.name not in aliases and right_spec.name not in aliases:
                    continue
                if (
                    left.numel()
                    and right.numel()
                    and left.data_ptr() < right.data_ptr() + right.numel() * right.element_size()
                    and right.data_ptr() < left.data_ptr() + left.numel() * left.element_size()
                ):
                    raise CanneryPackageError("Mutable input storage overlaps another input")
        by_name = {spec.name: tensor for spec, tensor in zip(selected.inputs, inputs, strict=True)}
        with torch.npu.device(self._device):
            outputs = {
                spec.name: by_name[spec.alias]
                if spec.alias is not None
                else torch_npu.empty_with_format(
                    spec.shape,
                    dtype=getattr(torch, spec.dtype),
                    device=inputs[0].device,
                    acl_format=2,
                )
                for spec in selected.outputs
            }
            input_descriptors = [
                tensor_descriptor(t.data_ptr(), spec.shape, spec.dtype)
                for t, spec in zip(inputs, selected.inputs, strict=True)
            ]
            output_descriptors = [
                tensor_descriptor(outputs[spec.name].data_ptr(), spec.shape, spec.dtype)
                for spec in selected.outputs
            ]
            scalar_descriptors = [
                scalar_descriptor(s.kind, scalars[s.name]) for s in selected.scalars
            ]
            validate_argument_budget(input_descriptors, output_descriptors, scalar_descriptors)
            native_inputs = (TensorDescriptor * len(inputs))(*input_descriptors)
            native_outputs = (TensorDescriptor * len(outputs))(*output_descriptors)
            native_scalars = (ScalarDescriptor * len(scalar_descriptors))(*scalar_descriptors)
            stream = torch.npu.current_stream(self._device)
            try:
                status = self._launch(
                    stream.npu_stream,
                    len(inputs),
                    native_inputs,
                    len(outputs),
                    native_outputs,
                    len(native_scalars),
                    native_scalars,
                )
            finally:
                # A native error can follow a partial enqueue; retain every buffer.
                for tensor in [*inputs, *outputs.values()]:
                    tensor.record_stream(stream)
            if status != 0:
                raise CanneryPackageError(f"Tensor native launch failed: {status}")
            return outputs
