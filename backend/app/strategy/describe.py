from app.indicators import REGISTRY
from app.strategy.models import (
    Condition,
    ConstantOperand,
    Group,
    IndicatorOperand,
    PriceOperand,
    RangeSpec,
    Strategy,
)

OPS = {">": ">", "<": "<", "crosses_above": "crosses above", "crosses_below": "crosses below"}


def _num(v) -> str:
    if isinstance(v, RangeSpec):
        return f"{v.min:g}–{v.max:g}"
    return f"{float(v):g}"


def operand_text(op) -> str:
    if isinstance(op, PriceOperand):
        return op.field.capitalize()
    if isinstance(op, ConstantOperand):
        return _num(op.value)
    if isinstance(op, IndicatorOperand):
        spec = REGISTRY.get(op.id)
        if spec is None:
            return op.id.upper()
        values = [_num(op.params.get(p.name, p.default)) for p in spec.params]
        label = op.id.upper() + (f"({', '.join(values)})" if values else "")
        return label if op.output == op.id else f"{label} {op.output}"
    return "?"


def _condition_text(c: Condition) -> str:
    left = operand_text(c.left)
    if c.op in ("rising", "falling"):
        unit = "bar" if c.bars == 1 else "bars"
        return f"{left} is {c.op} for {c.bars} {unit}"
    return f"{left} {OPS[c.op]} {operand_text(c.right)}"


def group_text(g: Group, nested: bool = False) -> str:
    parts = [group_text(i, True) if isinstance(i, Group) else _condition_text(i) for i in g.items]
    text = f" {g.join} ".join(parts)
    return f"({text})" if nested and len(parts) > 1 else text


def describe(strategy: Strategy) -> str:
    text = f"Buy when {group_text(strategy.buy)}. Sell when {group_text(strategy.sell)}."
    extras = []
    if strategy.stop_loss:
        unit = "%" if strategy.stop_loss.kind == "percent" else "× ATR"
        extras.append(f"stop loss {_num(strategy.stop_loss.value)}{unit}")
    if strategy.take_profit:
        unit = "%" if strategy.take_profit.kind == "percent" else "× ATR"
        extras.append(f"take profit {_num(strategy.take_profit.value)}{unit}")
    if strategy.max_hold_bars:
        extras.append(f"exit after {strategy.max_hold_bars} bars")
    if strategy.allow_short:
        extras.append("sell signals also open shorts")
    if extras:
        text += " " + "; ".join(extras).capitalize() + "."
    return text
