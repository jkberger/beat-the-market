from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Category = Literal["trend", "momentum", "volatility", "volume"]


@dataclass(frozen=True)
class Param:
    name: str
    kind: Literal["int", "float"]
    default: float
    min: float
    max: float


@dataclass(frozen=True)
class IndicatorSpec:
    id: str
    name: str
    category: Category
    params: list[Param]
    outputs: list[str]
    overlay: bool
    fn: Callable[..., pd.DataFrame] = field(repr=False)


REGISTRY: dict[str, IndicatorSpec] = {}


def register(spec: IndicatorSpec) -> None:
    if spec.id in REGISTRY:
        raise ValueError(f"Indicator already registered: {spec.id}")
    REGISTRY[spec.id] = spec


def indicator(
    id: str,
    name: str,
    category: Category,
    outputs: list[str],
    overlay: bool,
    params: list[Param] | None = None,
):
    """Decorator form of register()."""

    def wrap(fn):
        register(IndicatorSpec(id, name, category, params or [], outputs, overlay, fn))
        return fn

    return wrap


def length(default: int, lo: int = 2, hi: int = 400, name: str = "length") -> Param:
    return Param(name, "int", default, lo, hi)


def resolve_params(spec: IndicatorSpec, params: dict | None) -> dict:
    params = dict(params or {})
    known = {p.name: p for p in spec.params}
    unknown = set(params) - set(known)
    if unknown:
        raise ValueError(f"Unknown parameter(s) for {spec.id}: {', '.join(sorted(unknown))}")
    resolved = {}
    for p in spec.params:
        value = params.get(p.name, p.default)
        if not (p.min <= value <= p.max):
            raise ValueError(f"{spec.name} {p.name} must be between {p.min:g} and {p.max:g}")
        resolved[p.name] = int(value) if p.kind == "int" else float(value)
    return resolved


def compute(indicator_id: str, bars: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    spec = REGISTRY.get(indicator_id)
    if spec is None:
        raise ValueError(f"Unknown indicator: {indicator_id}")
    out = spec.fn(bars, **resolve_params(spec, params))
    out = out.astype(float)
    out.columns = spec.outputs
    out.index = bars.index
    return out
