from app.strategy.models import RiskSettings

RISK_PRESETS: dict[str, RiskSettings] = {
    "conservative": RiskSettings(
        preset="conservative",
        position_pct=25,
        stop_atr_mult=2,
        max_drawdown_pct=10,
        objective="sortino",
    ),
    "balanced": RiskSettings(
        preset="balanced",
        position_pct=50,
        stop_atr_mult=3,
        max_drawdown_pct=20,
        objective="sharpe",
    ),
    "aggressive": RiskSettings(
        preset="aggressive",
        position_pct=100,
        stop_atr_mult=None,
        max_drawdown_pct=None,
        objective="cagr",
    ),
}
