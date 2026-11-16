"""Bounded native tensor/scalar descriptors, independent of PyTorch and compilers.

This module defines the tensor-v2 bridge contract; package admission is separate.
Descriptors borrow their backing allocations. Callers must retain storage through
stream completion, validate the admitted signature, and enforce mutation policy.
"""

import ctypes
import math
import struct

MAX_RANK = 4
MAX_ELEMENTS = 1048576
MAX_ARGUMENTS = 32
MAX_CALL_BYTES = 32 * 1024 * 1024
DTYPES = {
    "float32": (1, 4),
    "float16": (2, 2),
    "bfloat16": (3, 2),
    "int8": (4, 1),
    "int32": (5, 4),
    "int64": (6, 8),
    "bool": (7, 1),
}


class TensorDescriptor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("sizes", ctypes.c_uint64 * MAX_RANK),
        ("strides", ctypes.c_uint64 * MAX_RANK),
        ("rank", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
    ]


class ScalarDescriptor(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("bits", ctypes.c_uint64),
    ]


def tensor_descriptor(pointer: int, shape: tuple[int, ...], dtype: str) -> TensorDescriptor:
    """Describe a contiguous allocation; reject narrowing/wrapping before ctypes.

    Empty tensors may use a null address. Strided views are materialized by the
    caller before building a descriptor. Negative strides are not supported.
    """
    if (
        type(pointer) is not int
        or not 0 <= pointer < 2**64
        or type(shape) is not tuple
        or not 1 <= len(shape) <= MAX_RANK
        or any(type(d) is not int or not 0 <= d <= MAX_ELEMENTS for d in shape)
        or math.prod(shape) > MAX_ELEMENTS
        or type(dtype) is not str
        or dtype not in DTYPES
    ):
        raise ValueError("Invalid tensor descriptor")
    if math.prod(shape) and not pointer:
        raise ValueError("Nonempty tensor requires a data address")
    _, width = DTYPES[dtype]
    if pointer + math.prod(shape) * width > 2**64:
        raise ValueError("Tensor address range wraps")
    if pointer % width:
        raise ValueError("Data address must be aligned to its dtype")
    strides = [0] * MAX_RANK
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = stride
        stride *= max(shape[axis], 1)
    return TensorDescriptor(
        pointer,
        (ctypes.c_uint64 * MAX_RANK)(*shape),
        (ctypes.c_uint64 * MAX_RANK)(*strides),
        len(shape),
        DTYPES[dtype][0],
    )


def scalar_descriptor(kind: str, value: int | float | bool) -> ScalarDescriptor:
    """Encode bool, signed int64 or finite IEEE binary64 as raw little-endian bits."""
    if kind == "bool" and type(value) is bool:
        return ScalarDescriptor(1, 0, int(value))
    if kind == "int64" and type(value) is int and -(2**63) <= value < 2**63:
        return ScalarDescriptor(2, 0, value & (2**64 - 1))
    if kind == "float64" and type(value) is float and math.isfinite(value):
        return ScalarDescriptor(3, 0, struct.unpack("<Q", struct.pack("<d", value))[0])
    raise ValueError("Invalid typed scalar")


def validate_argument_budget(
    inputs: list[TensorDescriptor],
    outputs: list[TensorDescriptor],
    scalars: list[ScalarDescriptor],
) -> None:
    """Bound counts and total contiguous tensor bytes before invoking native code.

    Accept only descriptors made by the checked factories, including canonical
    unused slots. This also detects accidental mutation of ctypes fields.
    """
    if not 1 <= len(inputs) <= MAX_ARGUMENTS or not 1 <= len(outputs) <= MAX_ARGUMENTS:
        raise ValueError("Expected one to 32 input and output tensors")
    if len(scalars) > MAX_ARGUMENTS:
        raise ValueError("Too many scalar arguments")
    total = 0
    dtype_names = {code: name for name, (code, _) in DTYPES.items()}
    for descriptor in [*inputs, *outputs]:
        if not isinstance(descriptor, TensorDescriptor) or descriptor.dtype not in dtype_names:
            raise ValueError("Invalid tensor descriptor")
        name = dtype_names[descriptor.dtype]
        expected = tensor_descriptor(
            descriptor.data or 0,
            tuple(descriptor.sizes[: descriptor.rank]),
            name,
        )
        if bytes(expected) != bytes(descriptor):
            raise ValueError("Noncanonical tensor descriptor")
        total += math.prod(descriptor.sizes[: descriptor.rank]) * DTYPES[name][1]
    if total > MAX_CALL_BYTES:
        raise ValueError("Call exceeds aggregate tensor byte budget")
    for scalar in scalars:
        if not isinstance(scalar, ScalarDescriptor) or scalar.reserved:
            raise ValueError("Invalid scalar descriptor")
        if scalar.kind == 1 and scalar.bits in (0, 1) or scalar.kind == 2:
            continue
        if scalar.kind == 3 and math.isfinite(
            struct.unpack("<d", struct.pack("<Q", scalar.bits))[0]
        ):
            continue
        raise ValueError("Invalid scalar descriptor")
