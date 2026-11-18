from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class RangeSpec(BaseModel):
    """A tunable value: min..max inclusive, by step."""

    min: float
    max: float
    step: float = Field(gt=0)

    @model_validator(mode="after")
    def _check(self):
        if self.max < self.min:
            raise ValueError("range max must be >= min")
        return self

    def values(self) -> list[float]:
        n = int(round((self.max - self.min) / self.step))
        vals = [round(self.min + i * self.step, 10) for i in range(n + 1)]
        return [v for v in vals if v <= self.max + 1e-9]


ParamValue = float | RangeSpec


class PriceOperand(BaseModel):
    kind: Literal["price"] = "price"
    field: Literal["open", "high", "low", "close"]


class IndicatorOperand(BaseModel):
    kind: Literal["indicator"] = "indicator"
    id: str
    params: dict[str, ParamValue] = {}
    output: str


class ConstantOperand(BaseModel):
    kind: Literal["constant"] = "constant"
    value: ParamValue


Operand = Annotated[PriceOperand | IndicatorOperand | ConstantOperand, Field(discriminator="kind")]

Op = Literal[">", "<", "crosses_above", "crosses_below", "rising", "falling"]
NEEDS_RIGHT = {">", "<", "crosses_above", "crosses_below"}


class Condition(BaseModel):
    kind: Literal["condition"] = "condition"
    left: Operand
    op: Op
    right: Operand | None = None
    bars: int = Field(default=1, ge=1, le=100)

    @model_validator(mode="after")
    def _check(self):
        if self.op in NEEDS_RIGHT and self.right is None:
            raise ValueError(f"condition '{self.op}' needs a right-hand operand")
        return self


class Group(BaseModel):
    kind: Literal["group"] = "group"
    join: Literal["and", "or"]
    items: list[Annotated[Condition | Group, Field(discriminator="kind")]] = Field(min_length=1)


class StopSpec(BaseModel):
    kind: Literal["percent", "atr"]
    value: ParamValue


class Strategy(BaseModel):
    name: str = "Untitled strategy"
    buy: Group
    sell: Group
    stop_loss: StopSpec | None = None
    take_profit: StopSpec | None = None
    max_hold_bars: int | None = Field(default=None, ge=1)
    allow_short: bool = False


class RiskSettings(BaseModel):
    preset: Literal["conservative", "balanced", "aggressive", "custom"] = "balanced"
    position_pct: float = Field(default=50, gt=0, le=100)
    stop_atr_mult: float | None = Field(default=3, gt=0)
    max_drawdown_pct: float | None = Field(default=20, gt=0, lt=100)
    standaside_bars: int = Field(default=20, ge=0)
    objective: Literal["sortino", "sharpe", "cagr"] = "sharpe"


class BacktestConfig(BaseModel):
    initial_equity: float = Field(default=10_000, gt=0)
    slippage_bps: float = Field(default=0.0, ge=0, le=500)


Group.model_rebuild()
