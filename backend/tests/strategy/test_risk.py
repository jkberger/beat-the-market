from app.strategy.risk import RISK_PRESETS


def test_presets_values():
    c, b, a = RISK_PRESETS["conservative"], RISK_PRESETS["balanced"], RISK_PRESETS["aggressive"]
    assert (c.position_pct, c.stop_atr_mult, c.max_drawdown_pct, c.objective) == (
        25,
        2,
        10,
        "sortino",
    )
    assert (b.position_pct, b.stop_atr_mult, b.max_drawdown_pct, b.objective) == (
        50,
        3,
        20,
        "sharpe",
    )
    assert (a.position_pct, a.stop_atr_mult, a.max_drawdown_pct, a.objective) == (
        100,
        None,
        None,
        "cagr",
    )
    assert c.standaside_bars == b.standaside_bars == 20
