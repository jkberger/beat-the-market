from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.serialize import clean
from app.data.providers import PriceProvider
from app.data.universe import get_price_provider, load_universe
from app.indicators import REGISTRY, compute
from app.strategy.risk import RISK_PRESETS

router = APIRouter(prefix="/api")


@router.get("/universe")
def universe() -> dict:
    return load_universe()


@router.get("/bars/{symbol}")
def bars(
    symbol: str,
    start: date | None = None,
    end: date | None = None,
    provider: PriceProvider = Depends(get_price_provider),
) -> dict:
    symbol = symbol.upper()
    df = provider.get_daily_bars(symbol, start, end)
    rows = [
        {
            "date": d.date().isoformat(),
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        }
        for d, r in df.iterrows()
    ]
    return {"symbol": symbol, "bars": clean(rows)}


@router.get("/indicators")
def indicators() -> list[dict]:
    return [
        {
            "id": s.id,
            "name": s.name,
            "category": s.category,
            "outputs": s.outputs,
            "overlay": s.overlay,
            "params": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "default": int(p.default) if p.kind == "int" else p.default,
                    "min": int(p.min) if p.kind == "int" else p.min,
                    "max": int(p.max) if p.kind == "int" else p.max,
                }
                for p in s.params
            ],
        }
        for s in REGISTRY.values()
    ]


class ComputeRequest(BaseModel):
    symbol: str
    id: str
    params: dict[str, float] = {}
    start: date | None = None
    end: date | None = None


@router.post("/indicators/compute")
def compute_indicator(
    req: ComputeRequest, provider: PriceProvider = Depends(get_price_provider)
) -> dict:
    df = provider.get_daily_bars(req.symbol.upper())
    out = compute(req.id, df, req.params)
    if req.start:
        out = out[out.index >= pd.Timestamp(req.start)]
    if req.end:
        out = out[out.index <= pd.Timestamp(req.end)]
    return {
        "dates": [d.date().isoformat() for d in out.index],
        "series": {c: clean(out[c].tolist()) for c in out.columns},
    }


@router.get("/risk-presets")
def risk_presets() -> dict:
    return {k: v.model_dump() for k, v in RISK_PRESETS.items()}
