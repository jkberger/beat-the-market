import statistics
from datetime import date
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import repo
from app.api.serialize import clean, series_points
from app.backtest.metrics import KEYS, buy_and_hold, compute_metrics
from app.backtest.validation import (
    NotEnoughHistory,
    SegmentResult,
    run_walk_forward,
    score_holdout,
    simple_segment,
)
from app.data.providers import DataSourceError, PriceProvider, SymbolNotFound
from app.data.universe import get_price_provider
from app.db import get_session
from app.strategy.describe import describe
from app.strategy.grid import _find_ranges, expand_grid, structure_key
from app.strategy.models import BacktestConfig, RiskSettings, Strategy

router = APIRouter(prefix="/api/backtests")
BENCHMARK = "SPY"
MAX_WORK = 400  # tuning combinations x symbols per request, keeps one request to ~2 min


class BacktestRequest(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=10)
    strategy: Strategy
    risk: RiskSettings = RiskSettings()
    config: BacktestConfig = BacktestConfig()
    mode: Literal["simple", "walk_forward"] = "simple"
    start: date | None = None
    end: date | None = None
    score_holdout: bool = False
    # Walk-forward training span: "all" = all history before each test, "recent" = last 3 years.
    training: Literal["all", "recent"] = "all"


def _range(df: pd.DataFrame, start: date | None, end: date | None) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def _baseline(bars: pd.DataFrame | None, config: BacktestConfig, start: date, end: date):
    if bars is None:
        return None
    sliced = _range(bars, start, end)
    if len(sliced) < 2:
        return None
    r = buy_and_hold(sliced, config)
    return {
        "equity": series_points(r.equity),
        "metrics": compute_metrics(r.equity, r.trades, r.position),
    }


def _segment_json(seg: SegmentResult, bars, spy, config) -> dict:
    stock = _baseline(bars, config, seg.start, seg.end)
    bench = _baseline(spy, config, seg.start, seg.end)
    return {
        "label": seg.label,
        "start": seg.start,
        "end": seg.end,
        "equity": series_points(seg.equity),
        "trades": seg.trades,
        "metrics": seg.metrics,
        "baseline_metrics": {
            "stock": stock["metrics"] if stock else None,
            "spy": bench["metrics"] if bench else None,
        },
    }


def _markers(segments: list[SegmentResult]) -> list[dict]:
    out = []
    for seg in segments:
        for t in seg.trades:
            entry_side = "buy" if t.direction == "long" else "sell"
            exit_side = "sell" if t.direction == "long" else "buy"
            out.append({"date": t.entry_date, "side": entry_side, "price": t.entry_price})
            out.append(
                {
                    "date": t.exit_date,
                    "side": exit_side,
                    "price": t.exit_price,
                    "reason": t.exit_reason,
                }
            )
    return sorted(out, key=lambda m: m["date"])


def _run_symbol(req: BacktestRequest, symbol: str, bars, spy, s: Session) -> dict:
    windows: list[dict] = []
    count = None
    if req.mode == "simple":
        if list(_find_ranges(req.strategy.model_dump())):
            expand_grid(req.strategy)  # a ValueError here carries the real limit message
            raise HTTPException(
                422, "Simple mode can't tune ranges. Switch to walk-forward to tune them."
            )
        try:
            segments = [
                simple_segment(
                    req.strategy,
                    _range(bars, None, req.end),
                    req.risk,
                    req.config,
                    req.start,
                    req.end,
                )
            ]
        except NotEnoughHistory as e:
            raise HTTPException(422, f"{symbol}: {e}") from e
    else:
        holdout_start = repo.get_holdout_start(s)
        in_range = _range(bars, req.start, None)
        try:
            oos, windows, final = run_walk_forward(
                req.strategy,
                in_range,
                req.risk,
                req.config,
                holdout_start,
                anchored=req.training == "all",
            )
        except NotEnoughHistory as e:
            raise HTTPException(422, f"{symbol}: {e}") from e
        segments = [oos]
        key = structure_key(req.strategy)
        if req.score_holdout:
            try:
                segments.append(score_holdout(final, in_range, req.risk, req.config, holdout_start))
            except NotEnoughHistory as e:
                raise HTTPException(422, f"{symbol}: {e}") from e
            count = repo.record_holdout_score(s, key, symbol)
        else:
            count = repo.holdout_count(s, key, symbol)

    span_start, span_end = segments[0].start, segments[-1].end
    return {
        "symbol": symbol,
        "mode": req.mode,
        "strategy_text": describe(req.strategy),
        "risk": req.risk.model_dump(),
        "config": req.config.model_dump(),
        "segments": {seg.label: _segment_json(seg, bars, spy, req.config) for seg in segments},
        "baselines": {
            "stock": _baseline(bars, req.config, span_start, span_end),
            "spy": _baseline(spy, req.config, span_start, span_end),
        },
        "markers": _markers(segments),
        "windows": windows,
        "training": req.training if req.mode == "walk_forward" else None,
        "holdout_count": count,
    }


def _summary(runs: list[dict]) -> dict:
    per_symbol = []
    for run in runs:
        primary = run["segments"].get("out_of_sample") or run["segments"]["backtest"]
        per_symbol.append({"symbol": run["symbol"], "metrics": primary["metrics"]})
    median = {}
    for k in KEYS:
        vals = [p["metrics"][k] for p in per_symbol if p["metrics"][k] is not None]
        median[k] = statistics.median(vals) if vals else None
    return {"median": median, "per_symbol": per_symbol}


@router.post("")
def run_backtest(
    req: BacktestRequest,
    provider: PriceProvider = Depends(get_price_provider),
    s: Session = Depends(get_session),
) -> dict:
    symbols = list(dict.fromkeys(sym.upper().strip() for sym in req.symbols))
    if req.start and req.end and req.start >= req.end:
        raise HTTPException(422, "The start date must be before the end date.")
    if req.mode == "walk_forward":
        work = len(expand_grid(req.strategy)) * len(symbols)
        if work > MAX_WORK:
            raise HTTPException(
                422,
                f"That's {work:,} tuning combinations across {len(symbols)} stocks; the limit "
                f"is {MAX_WORK} per run. Use fewer stocks, narrower ranges or bigger steps.",
            )
    try:
        spy = provider.get_daily_bars(BENCHMARK)
    except (SymbolNotFound, DataSourceError):
        spy = None
    runs = [_run_symbol(req, sym, provider.get_daily_bars(sym), spy, s) for sym in symbols]
    body = clean(
        {
            "symbol": ", ".join(symbols),
            "mode": req.mode,
            "strategy_name": req.strategy.name,
            "request": req.model_dump(mode="json"),
            "summary": _summary(runs),
            "runs": runs,
        }
    )
    run_id = repo.save_run(s, body)
    return {"id": run_id, **body}


@router.get("")
def list_backtests(s: Session = Depends(get_session)) -> list[dict]:
    return repo.list_runs(s)


@router.get("/{run_id}")
def get_backtest(run_id: int, s: Session = Depends(get_session)) -> dict:
    run = repo.get_run(s, run_id)
    if run is None:
        raise HTTPException(404, "Backtest not found")
    return run
