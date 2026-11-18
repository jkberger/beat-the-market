from app.strategy.models import Strategy


def rsi_strategy(buy_level=30, sell_level=70, length=14, **extra) -> Strategy:
    return Strategy.model_validate(
        {
            "name": "RSI swing",
            "buy": {
                "kind": "group",
                "join": "and",
                "items": [
                    {
                        "kind": "condition",
                        "left": {
                            "kind": "indicator",
                            "id": "rsi",
                            "params": {"length": length},
                            "output": "rsi",
                        },
                        "op": "crosses_above",
                        "right": {"kind": "constant", "value": buy_level},
                    }
                ],
            },
            "sell": {
                "kind": "group",
                "join": "and",
                "items": [
                    {
                        "kind": "condition",
                        "left": {
                            "kind": "indicator",
                            "id": "rsi",
                            "params": {"length": length},
                            "output": "rsi",
                        },
                        "op": "crosses_below",
                        "right": {"kind": "constant", "value": sell_level},
                    }
                ],
            },
            **extra,
        }
    )


def _sma(length):
    return {"kind": "indicator", "id": "sma", "params": {"length": length}, "output": "sma"}


def golden_cross_strategy(fast=50, slow=200, **extra) -> Strategy:
    """Buy when the fast SMA crosses above the slow one, sell on the reverse cross."""

    def side(op):
        return {
            "kind": "group",
            "join": "and",
            "items": [{"kind": "condition", "left": _sma(fast), "op": op, "right": _sma(slow)}],
        }

    return Strategy.model_validate(
        {
            "name": "Golden cross",
            "buy": side("crosses_above"),
            "sell": side("crosses_below"),
            **extra,
        }
    )
