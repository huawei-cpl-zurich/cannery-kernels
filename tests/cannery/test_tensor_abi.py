import ctypes
import math
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from cannery_kernels.tensor_abi import (
    DTYPES,
    ScalarDescriptor,
    scalar_descriptor,
    tensor_descriptor,
    validate_argument_budget,
)


class TensorAbiTests(unittest.TestCase):
    def test_contiguous_shape_stride_and_empty(self):
        tensor = tensor_descriptor(4096, (2, 3, 4), "bfloat16")
        self.assertEqual(list(tensor.sizes), [2, 3, 4, 0])
        self.assertEqual(list(tensor.strides), [12, 4, 1, 0])
        self.assertEqual((tensor.rank, tensor.dtype), (3, 3))
        empty = tensor_descriptor(0, (0, 3), "float32")
        validate_argument_budget([empty], [tensor], [])
        for dtype in DTYPES:
            validate_argument_budget(
                [tensor_descriptor(4096, (1,), dtype)], [tensor], []
            )

    def test_invalid_tensor_addresses_and_shapes(self):
        for pointer, shape, dtype in [
            (True, (1,), "int8"),
            (-1, (1,), "int8"),
            (2**64, (1,), "int8"),
            (0, (1,), "int8"),
            (4097, (1,), "int64"),
            (2**64 - 8, (2,), "int64"),
            (4096, (), "int8"),
            (4096, (1,) * 5, "int8"),
            (4096, (True,), "int8"),
            (4096, (-1,), "int8"),
            (4096, (1024, 1025), "int8"),
            (4096, (1,), "float64"),
            (4096, [1], "int8"),
        ]:
            with self.subTest(pointer=pointer, shape=shape, dtype=dtype):
                with self.assertRaises(ValueError):
                    tensor_descriptor(pointer, shape, dtype)

    def test_typed_scalar_bits_and_invalid_values(self):
        self.assertEqual(scalar_descriptor("int64", -1).bits, 2**64 - 1)
        self.assertEqual(scalar_descriptor("int64", -(2**63)).bits, 2**63)
        self.assertEqual(scalar_descriptor("bool", True).bits, 1)
        negative_zero = scalar_descriptor("float64", -0.0)
        self.assertEqual(negative_zero.bits, 2**63)
        self.assertEqual(
            struct.unpack("<d", struct.pack("<Q", negative_zero.bits))[0], -0.0
        )
        for kind, value in [
            ("bool", 1),
            ("int64", True),
            ("int64", 2**63),
            ("float64", 1),
            ("float64", math.inf),
            ("float64", math.nan),
            ("str", "x"),
        ]:
            with self.assertRaises(ValueError):
                scalar_descriptor(kind, value)

    def test_corruption_and_aggregate_bounds_before_native_call(self):
        tensor = tensor_descriptor(4096, (1048576,), "int64")
        validate_argument_budget([tensor], [tensor], [scalar_descriptor("bool", False)])
        with self.assertRaisesRegex(ValueError, "aggregate"):
            validate_argument_budget([tensor] * 4, [tensor], [])
        for inputs, outputs, scalars in [
            ([], [tensor], []),
            ([tensor] * 33, [tensor], []),
            ([tensor], [], []),
            ([tensor], [tensor], [scalar_descriptor("bool", False)] * 33),
            ([tensor], [tensor], [ScalarDescriptor(1, 0, 2)]),
            ([tensor], [tensor], [ScalarDescriptor(2, 1, 0)]),
            ([tensor], [tensor], [ScalarDescriptor(3, 0, 0x7FF0000000000000)]),
            ([tensor], [tensor], [ScalarDescriptor(4, 0, 0)]),
        ]:
            with self.assertRaises(ValueError):
                validate_argument_budget(inputs, outputs, scalars)
        tensor.strides[0] = 2
        with self.assertRaisesRegex(ValueError, "Noncanonical"):
            validate_argument_budget([tensor], [tensor], [])

    @unittest.skipUnless(shutil.which("cc"), "C compiler required for ABI layout test")
    def test_native_layout_matches_python(self):
        import cannery_kernels.tensor_abi as abi

        assertions = []
        for native, python in [
            ("CanneryTensorV2", abi.TensorDescriptor),
            ("CanneryScalarV2", abi.ScalarDescriptor),
        ]:
            assertions.append(
                f'_Static_assert(sizeof({native}) == {ctypes.sizeof(python)}, "size");'
            )
            for field, _ in python._fields_:
                offset = getattr(python, field).offset
                assertions.append(
                    f'_Static_assert(offsetof({native}, {field}) == {offset}, "offset");'
                )
        source = '#include <stddef.h>\n#include "tensor_abi.h"\n' + "\n".join(
            assertions
        )
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    "-Werror",
                    "-x",
                    "c",
                    "-c",
                    "-",
                    "-I",
                    str(Path(abi.__file__).parent),
                    "-o",
                    str(Path(directory) / "abi.o"),
                ],
                input=source,
                text=True,
                check=True,
                capture_output=True,
                timeout=30,
            )
