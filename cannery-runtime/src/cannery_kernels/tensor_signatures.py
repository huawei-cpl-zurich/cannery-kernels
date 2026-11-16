"""Validate static tensor-v2 call domains without loading native code."""

import json
import re
from dataclasses import dataclass
from typing import Any

from .tensor_abi import scalar_descriptor, tensor_descriptor, validate_argument_budget


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    alias: str | None = None


@dataclass(frozen=True)
class ScalarSpec:
    name: str
    kind: str
    value: int | float | bool


@dataclass(frozen=True)
class TensorVariant:
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    scalars: tuple[ScalarSpec, ...]


def _name(value: Any) -> str:
    if type(value) is not str or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) is None:
        raise ValueError("Invalid argument name")
    return value


def _tensors(value: Any, output: bool) -> tuple[TensorSpec, ...]:
    if type(value) is not list or not 1 <= len(value) <= 32:
        raise ValueError("Expected one to 32 tensor arguments")
    result = []
    for item in value:
        keys = {"name", "shape", "dtype"} | ({"alias"} if output else set())
        if type(item) is not dict or set(item) != keys or type(item["shape"]) is not list:
            raise ValueError("Invalid tensor signature")
        shape = tuple(item["shape"])
        tensor_descriptor(4096, shape, item["dtype"])
        alias = item.get("alias")
        if alias is not None:
            alias = _name(alias)
        result.append(TensorSpec(_name(item["name"]), shape, item["dtype"], alias))
    if len({item.name for item in result}) != len(result):
        raise ValueError("Duplicate tensor argument name")
    return tuple(result)


def parse_tensor_variants(value: Any) -> tuple[TensorVariant, ...]:
    """Reject ambiguity, undeclared mutations and unbounded call signatures.

    Output aliases declare mutable inputs, always with matching shape and dtype.
    All other inputs are read-only. Aliases are returned for numerical checking.
    Each variant has the same named inputs/outputs/scalars and exact scalar kinds.
    Input tensor shapes/dtypes and scalar bit patterns uniquely select a variant.
    """
    if type(value) is not list or not 1 <= len(value) <= 32:
        raise ValueError("Expected one to 32 static variants")
    variants, keys = [], set()
    interface = None
    for record in value:
        if type(record) is not dict or set(record) != {"inputs", "outputs", "scalars"}:
            raise ValueError("Invalid variant record")
        inputs, outputs = _tensors(record["inputs"], False), _tensors(record["outputs"], True)
        scalars = record["scalars"]
        if type(scalars) is not list or len(scalars) > 32:
            raise ValueError("Expected up to 32 typed scalars")
        parsed = []
        for item in scalars:
            if type(item) is not dict or set(item) != {"name", "kind", "value"}:
                raise ValueError("Invalid scalar signature")
            scalar_descriptor(item["kind"], item["value"])
            parsed.append(ScalarSpec(_name(item["name"]), item["kind"], item["value"]))
        scalar_specs = tuple(parsed)
        if len({item.name for item in inputs} | {s.name for s in parsed}) != len(inputs) + len(
            parsed
        ):
            raise ValueError("Duplicate input/scalar argument name")
        aliases = set()
        by_name = {item.name: item for item in inputs}
        for output in outputs:
            if output.alias is None:
                continue
            original = by_name.get(output.alias)
            if (
                original is None
                or original.shape != output.shape
                or original.dtype != output.dtype
                or output.alias in aliases
            ):
                raise ValueError("Output alias must identify one matching mutable input")
            aliases.add(output.alias)
        validate_argument_budget(
            [tensor_descriptor(4096, i.shape, i.dtype) for i in inputs],
            [tensor_descriptor(4096, i.shape, i.dtype) for i in outputs],
            [scalar_descriptor(s.kind, s.value) for s in scalar_specs],
        )
        current_interface = (
            tuple(i.name for i in inputs),
            tuple(i.name for i in outputs),
            tuple((s.name, s.kind) for s in scalar_specs),
        )
        if interface is not None and interface != current_interface:
            raise ValueError("Static variants must share an ordered named interface")
        interface = current_interface
        key = json.dumps(
            {
                "inputs": [(i.shape, i.dtype) for i in inputs],
                "scalars": [bytes(scalar_descriptor(s.kind, s.value)).hex() for s in scalar_specs],
            },
            sort_keys=True,
        )
        if key in keys:
            raise ValueError("Duplicate static dispatch domain")
        keys.add(key)
        variants.append(TensorVariant(inputs, outputs, scalar_specs))
    return tuple(variants)
