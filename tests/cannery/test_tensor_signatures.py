import copy
import hashlib
import unittest
from unittest.mock import patch

from test_package import package

import cannery_kernels as loader
from cannery_kernels.tensor_signatures import parse_tensor_variants


def variant():
    return {
        "inputs": [
            {"name": "x", "shape": [2, 3], "dtype": "float16"},
            {"name": "cache", "shape": [4, 3], "dtype": "float16"},
        ],
        "outputs": [
            {
                "name": "cache_out",
                "shape": [4, 3],
                "dtype": "float16",
                "alias": "cache",
            },
            {"name": "indices", "shape": [2], "dtype": "int32", "alias": None},
        ],
        "scalars": [{"name": "offset", "kind": "int64", "value": 1}],
    }


class SignatureTests(unittest.TestCase):
    def test_named_static_domains_and_mutation_alias(self):
        record = variant()
        other = copy.deepcopy(record)
        other["scalars"][0]["value"] = 2
        parsed = parse_tensor_variants([record, other])
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].outputs[0].alias, "cache")
        self.assertEqual(parsed[1].scalars[0].value, 2)
        with self.assertRaisesRegex(ValueError, "Duplicate static"):
            parse_tensor_variants([record, record])

    def test_invalid_records_and_ambiguous_interfaces(self):
        for mutate in [
            lambda v: v.update(extra=1),
            lambda v: v.update(inputs=[]),
            lambda v: v["inputs"][0].update(shape=[1024, 1025]),
            lambda v: v["inputs"][0].update(name="../x"),
            lambda v: v["inputs"][0].update(dtype="float64"),
            lambda v: v["inputs"][0].update(alias="cache"),
            lambda v: v["outputs"][0].update(alias="unknown"),
            lambda v: v["outputs"][0].update(shape=[2, 3]),
            lambda v: v["outputs"][0].update(dtype="int8"),
            lambda v: v["outputs"][1].update(name="cache_out"),
            lambda v: v["scalars"][0].update(name="x"),
            lambda v: v["scalars"][0].update(value=True),
            lambda v: v.update(scalars=v["scalars"] * 33),
        ]:
            record = variant()
            mutate(record)
            with self.assertRaises(ValueError):
                parse_tensor_variants([record])
        other = variant()
        other["inputs"].reverse()
        with self.assertRaisesRegex(ValueError, "ordered named interface"):
            parse_tensor_variants([variant(), other])

    def test_float_bit_dispatch_preserves_signed_zero(self):
        record = variant()
        record["scalars"] = [{"name": "epsilon", "kind": "float64", "value": 0.0}]
        other = copy.deepcopy(record)
        other["scalars"][0]["value"] = -0.0
        self.assertEqual(len(parse_tensor_variants([record, other])), 2)

    def test_package_reader_verifies_but_does_not_execute_tensor_packages(self):
        def tensor(metadata):
            metadata.update(schema_version=3, abi="tensor-v2", variants=[variant()])

        data, sha, metadata = package(tensor)
        self.assertEqual(sha, hashlib.sha256(data).hexdigest())
        self.assertEqual(loader.verify_cannery_package(data, sha)[0], metadata)
        with patch.object(
            loader.ctypes, "CDLL", side_effect=AssertionError("must not execute")
        ):
            with self.assertRaisesRegex(
                loader.CanneryPackageError, "execution is not enabled"
            ):
                loader.load_cannery_package(data, sha, trust_remote_code=True)

        def a3(metadata):
            tensor(metadata)
            metadata.update(device="a3")
            metadata["runtime"]["soc"] = "Ascend910B4"

        data, sha, _ = package(a3)
        with self.assertRaises(loader.CanneryPackageError):
            loader.verify_cannery_package(data, sha)
