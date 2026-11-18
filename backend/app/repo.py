from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import BacktestRun, HoldoutScore, SavedStrategy, Setting
from app.strategy.grid import structure_key
from app.strategy.models import Strategy

HOLDOUT_KEY = "holdout_start"


def _add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # Feb 29
        return d.replace(year=d.year + years, day=28)


def get_holdout_start(s: Session) -> date:
    row = s.get(Setting, HOLDOUT_KEY)
    if row is None:
        row = Setting(key=HOLDOUT_KEY, value=_add_years(date.today(), -2).isoformat())
        s.add(row)
        s.commit()
    return date.fromisoformat(row.value)


def set_holdout_start(s: Session, d: date) -> None:
    row = s.get(Setting, HOLDOUT_KEY)
    if row is None:
        s.add(Setting(key=HOLDOUT_KEY, value=d.isoformat()))
    else:
        row.value = d.isoformat()
    s.commit()


def save_strategy(s: Session, strategy: Strategy) -> int:
    row = SavedStrategy(
        name=strategy.name,
        body=strategy.model_dump(mode="json"),
        structure_key=structure_key(strategy),
    )
    s.add(row)
    s.commit()
    return row.id


def _strategy_row(row: SavedStrategy) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "strategy": row.body,
        "created_at": row.created_at.isoformat(),
    }


def list_strategies(s: Session) -> list[dict]:
    rows = s.scalars(select(SavedStrategy).order_by(SavedStrategy.id.desc()))
    return [_strategy_row(r) for r in rows]


def get_strategy(s: Session, strategy_id: int) -> Strategy | None:
    row = s.get(SavedStrategy, strategy_id)
    return None if row is None else Strategy.model_validate(row.body)


def delete_strategy(s: Session, strategy_id: int) -> bool:
    row = s.get(SavedStrategy, strategy_id)
    if row is None:
        return False
    s.delete(row)
    s.commit()
    return True


def holdout_count(s: Session, key: str, symbol: str) -> int:
    row = s.get(HoldoutScore, (key, symbol))
    return 0 if row is None else row.count


def record_holdout_score(s: Session, key: str, symbol: str) -> int:
    row = s.get(HoldoutScore, (key, symbol))
    if row is None:
        row = HoldoutScore(structure_key=key, symbol=symbol, count=0)
        s.add(row)
    row.count += 1
    s.commit()
    return row.count


def save_run(s: Session, body: dict) -> int:
    row = BacktestRun(
        symbol=body.get("symbol", ""),
        mode=body.get("mode", ""),
        strategy_name=body.get("strategy_name", ""),
        body=body,
    )
    s.add(row)
    s.commit()
    return row.id


def list_runs(s: Session, limit: int = 20) -> list[dict]:
    rows = s.scalars(select(BacktestRun).order_by(BacktestRun.id.desc()).limit(limit))
    return [
        {
            "id": r.id,
            "symbol": r.symbol,
            "mode": r.mode,
            "strategy_name": r.strategy_name,
            "created_at": r.created_at.isoformat(),
            "summary": r.body.get("summary"),
        }
        for r in rows
    ]


def get_run(s: Session, run_id: int) -> dict | None:
    row = s.get(BacktestRun, run_id)
    return None if row is None else {**row.body, "id": row.id}
