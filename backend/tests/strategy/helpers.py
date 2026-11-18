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
