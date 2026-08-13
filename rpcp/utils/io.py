"""Config (de)serialisation helpers: YAML/JSON <-> nested dataclasses."""

from __future__ import annotations

import json
import types
import typing
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import yaml

__all__ = [
    "load_yaml",
    "save_yaml",
    "from_dict",
    "to_dict",
    "save_json",
    "load_json",
    "deep_update",
    "tensor_to_list",
]

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping at the top level of {path}, got {type(data)}")
    return data


def save_yaml(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_dict(obj) if is_dataclass(obj) else obj
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_dict(obj) if is_dataclass(obj) else obj
    path.write_text(json.dumps(payload, indent=2, default=tensor_to_list), encoding="utf-8")


def tensor_to_list(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``base``."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _is_optional(annotation: Any) -> bool:
    origin = typing.get_origin(annotation)
    return origin in (typing.Union, types.UnionType) and type(None) in typing.get_args(annotation)


def _unwrap_optional(annotation: Any) -> Any:
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    return args[0] if len(args) == 1 else annotation


def from_dict(cls: type[T], payload: dict[str, Any] | None) -> T:
    """Instantiate a (possibly nested) dataclass from a plain mapping.

    Unknown keys raise a ``TypeError`` so that config typos fail loudly instead
    of silently doing nothing.
    """
    payload = dict(payload or {})
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")

    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise TypeError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for field_ in fields(cls):
        if field_.name not in payload:
            continue
        value = payload[field_.name]
        annotation = hints[field_.name]
        if _is_optional(annotation):
            if value is None:
                kwargs[field_.name] = None
                continue
            annotation = _unwrap_optional(annotation)
        if is_dataclass(annotation) and isinstance(value, dict):
            kwargs[field_.name] = from_dict(annotation, value)
        elif isinstance(value, list) and typing.get_origin(annotation) in (list, tuple):
            (inner,) = typing.get_args(annotation)[:1] or (Any,)
            if is_dataclass(inner):
                kwargs[field_.name] = [from_dict(inner, v) for v in value]
            elif typing.get_origin(annotation) is tuple:
                kwargs[field_.name] = tuple(value)
            else:
                kwargs[field_.name] = value
        else:
            kwargs[field_.name] = value

    for field_ in fields(cls):
        if (
            field_.name not in kwargs
            and field_.default is MISSING
            and field_.default_factory is MISSING  # type: ignore[misc]
        ):
            raise TypeError(f"Missing required config key '{field_.name}' for {cls.__name__}")
    return cls(**kwargs)  # type: ignore[return-value]


def to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses/tensors/paths into JSON-friendly types."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return tensor_to_list(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
