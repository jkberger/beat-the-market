import pandas as pd

from app.indicators import compute
from app.strategy.models import (
    Condition,
    ConstantOperand,
    Group,
    IndicatorOperand,
    PriceOperand,
    RangeSpec,
    Strategy,
)


def _concrete(value) -> float:
    if isinstance(value, RangeSpec):
        raise ValueError("Strategy still has a tunable range; expand or tune it first")
    return float(value)


def operand_series(operand, bars: pd.DataFrame, cache: dict) -> pd.Series:
    if isinstance(operand, PriceOperand):
        return bars[operand.field]
    if isinstance(operand, ConstantOperand):
        return pd.Series(_concrete(operand.value), index=bars.index)
    if isinstance(operand, IndicatorOperand):
        params = {k: _concrete(v) for k, v in operand.params.items()}
        key = (operand.id, tuple(sorted(params.items())))
        if key not in cache:
            cache[key] = compute(operand.id, bars, params)
        frame = cache[key]
        if operand.output not in frame.columns:
            raise ValueError(f"{operand.id} has no output '{operand.output}'")
        return frame[operand.output]
    raise ValueError(f"Unknown operand: {operand!r}")


def _condition(cond: Condition, bars: pd.DataFrame, cache: dict) -> pd.Series:
    left = operand_series(cond.left, bars, cache)
    if cond.op in ("rising", "falling"):
        step = left.diff()
        ok = step > 0 if cond.op == "rising" else step < 0
        return ok.astype(float).rolling(cond.bars).sum().eq(cond.bars)
    right = operand_series(cond.right, bars, cache)
    if cond.op == ">":
        return left > right
    if cond.op == "<":
        return left < right
    above, prev_above = left > right, left.shift(1) > right.shift(1)
    below, prev_below = left < right, left.shift(1) < right.shift(1)
    if cond.op == "crosses_above":
        return above & (prev_below | (left.shift(1) == right.shift(1)))
    return below & (prev_above | (left.shift(1) == right.shift(1)))


def evaluate_group(group: Group, bars: pd.DataFrame, cache: dict | None = None) -> pd.Series:
    cache = {} if cache is None else cache
    parts = [
        evaluate_group(item, bars, cache)
        if isinstance(item, Group)
        else _condition(item, bars, cache)
        for item in group.items
    ]
    out = parts[0]
    for p in parts[1:]:
        out = (out & p) if group.join == "and" else (out | p)
    return out.fillna(False).astype(bool)


def signals(strategy: Strategy, bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    cache: dict = {}
    return evaluate_group(strategy.buy, bars, cache), evaluate_group(strategy.sell, bars, cache)
