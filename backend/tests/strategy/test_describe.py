from app.strategy.describe import describe
from app.strategy.models import Strategy


def test_describe_matches_example():
    s = Strategy.model_validate(
        {
            "name": "x",
            "buy": {
                "kind": "group",
                "join": "and",
                "items": [
                    {
                        "kind": "condition",
                        "left": {
                            "kind": "indicator",
                            "id": "rsi",
                            "params": {"length": 14},
                            "output": "rsi",
                        },
                        "op": "crosses_above",
                        "right": {"kind": "constant", "value": 30},
                    },
                    {
                        "kind": "condition",
                        "left": {"kind": "price", "field": "close"},
                        "op": ">",
                        "right": {
                            "kind": "indicator",
                            "id": "sma",
                            "params": {"length": 200},
                            "output": "sma",
                        },
                    },
                ],
            },
            "sell": {
                "kind": "group",
                "join": "and",
                "items": [
                    {
                        "kind": "condition",
                        "left": {"kind": "indicator", "id": "rsi", "params": {}, "output": "rsi"},
                        "op": "crosses_below",
                        "right": {"kind": "constant", "value": 70},
                    },
                ],
            },
        }
    )
    assert describe(s) == (
        "Buy when RSI(14) crosses above 30 and Close > SMA(200). "
        "Sell when RSI(14) crosses below 70."
    )


def test_describe_multi_output_and_range():
    s = Strategy.model_validate(
        {
            "name": "x",
            "buy": {
                "kind": "group",
                "join": "or",
                "items": [
                    {
                        "kind": "condition",
                        "left": {"kind": "indicator", "id": "macd", "params": {}, "output": "macd"},
                        "op": "crosses_above",
                        "right": {
                            "kind": "indicator",
                            "id": "macd",
                            "params": {},
                            "output": "signal",
                        },
                    },
                    {
                        "kind": "condition",
                        "left": {"kind": "price", "field": "close"},
                        "op": "rising",
                        "bars": 3,
                    },
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
                            "params": {"length": {"min": 7, "max": 21, "step": 7}},
                            "output": "rsi",
                        },
                        "op": ">",
                        "right": {"kind": "constant", "value": 70},
                    },
                ],
            },
        }
    )
    assert describe(s) == (
        "Buy when MACD(12, 26, 9) crosses above MACD(12, 26, 9) signal or Close is rising "
        "for 3 bars. Sell when RSI(7–21) > 70."
    )
