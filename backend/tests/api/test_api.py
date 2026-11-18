import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.data.providers import SymbolNotFound, slice_bars
from app.data.universe import get_price_provider
from app.db import Base, get_session
from app.main import create_app
from tests.conftest import load_fixture_bars
from tests.strategy.helpers import rsi_strategy


class FixtureProvider:
    def get_daily_bars(self, symbol, start=None, end=None):
        symbol = symbol.upper()
        if symbol not in {"AAPL", "SPY", "MSFT"}:
            raise SymbolNotFound(symbol)
        df = load_fixture_bars()
        if symbol == "SPY":
            df[["open", "high", "low", "close", "adj_close"]] *= 3
        if symbol == "MSFT":
            df = df.iloc[::-1].set_axis(df.index)  # different path, same dates
        return slice_bars(df, start, end)


class ShortProvider(FixtureProvider):
    def get_daily_bars(self, symbol, start=None, end=None):
        return super().get_daily_bars(symbol).iloc[-504:]


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    maker = sessionmaker(engine)

    def session():
        with maker() as s:
            yield s

    app = create_app()
    app.dependency_overrides[get_session] = session
    app.dependency_overrides[get_price_provider] = lambda: FixtureProvider()
    c = TestClient(app)
    assert c.put("/api/settings", json={"holdout_start": "2021-09-01"}).status_code == 200
    c.app_ref = app
    return c


def _body(**over):
    body = {
        "symbols": ["AAPL"],
        "strategy": rsi_strategy().model_dump(mode="json"),
        "risk": {
            "preset": "aggressive",
            "position_pct": 100,
            "stop_atr_mult": None,
            "max_drawdown_pct": None,
            "objective": "cagr",
        },
        "config": {"initial_equity": 10000, "slippage_bps": 0},
        "mode": "simple",
    }
    body.update(over)
    return body


def test_unknown_symbol_404(client):
    r = client.get("/api/bars/AAPLL")
    assert r.status_code == 404
    assert r.json()["detail"] == "Unknown symbol: AAPLL"


def test_unknown_symbol_in_backtest_404(client):
    r = client.post("/api/backtests", json=_body(symbols=["AAPLL"]))
    assert r.status_code == 404


def test_bars(client):
    r = client.get("/api/bars/aapl", params={"start": "2020-01-01", "end": "2020-12-31"})
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "AAPL"
    assert data["bars"][0]["date"].startswith("2020-01")
    assert set(data["bars"][0]) == {"date", "open", "high", "low", "close", "volume"}


def test_universe(client):
    assert client.get("/api/universe").json()["quick_picks"][0] == "SPY"


def test_indicators_list_has_25(client):
    items = client.get("/api/indicators").json()
    assert len(items) == 25
    rsi = next(i for i in items if i["id"] == "rsi")
    assert rsi["params"][0] == {
        "name": "length",
        "kind": "int",
        "default": 14,
        "min": 2,
        "max": 200,
    }


def test_compute_indicator(client):
    r = client.post(
        "/api/indicators/compute", json={"symbol": "AAPL", "id": "sma", "params": {"length": 50}}
    )
    data = r.json()
    assert len(data["dates"]) == len(data["series"]["sma"])
    assert data["series"]["sma"][0] is None


def test_risk_presets(client):
    assert client.get("/api/risk-presets").json()["conservative"]["position_pct"] == 25


def test_describe(client):
    r = client.post("/api/strategies/describe", json=rsi_strategy().model_dump(mode="json"))
    assert r.json()["text"].startswith("Buy when RSI(14) crosses above 30")


def test_strategy_crud(client):
    sid = client.post("/api/strategies", json=rsi_strategy().model_dump(mode="json")).json()["id"]
    assert client.get(f"/api/strategies/{sid}").json()["strategy"]["name"] == "RSI swing"
    assert len(client.get("/api/strategies").json()) == 1
    assert client.delete(f"/api/strategies/{sid}").status_code == 204
    assert client.get(f"/api/strategies/{sid}").status_code == 404


def test_backtest_simple_has_baselines_and_markers(client):
    r = client.post("/api/backtests", json=_body())
    assert r.status_code == 200
    run = r.json()["runs"][0]
    assert set(run["segments"]) == {"backtest"}
    seg = run["segments"]["backtest"]
    assert seg["metrics"]["num_trades"] > 5
    assert seg["trades"][0]["exit_reason"] in {"signal", "stop", "target", "max_hold", "end"}
    assert run["baselines"]["stock"]["metrics"]["num_trades"] == 1
    assert run["baselines"]["spy"]["metrics"]["total_return"] == pytest.approx(
        run["baselines"]["stock"]["metrics"]["total_return"], rel=1e-6
    )
    assert {m["side"] for m in run["markers"]} == {"buy", "sell"}
    assert run["strategy_text"].startswith("Buy when")
    assert seg["baseline_metrics"]["stock"]["num_trades"] == 1
    assert client.get(f"/api/backtests/{r.json()['id']}").json()["runs"][0]["symbol"] == "AAPL"
    assert client.get("/api/backtests").json()[0]["id"] == r.json()["id"]


def test_simple_mode_rejects_ranges(client):
    s = rsi_strategy(buy_level={"min": 25, "max": 35, "step": 5}).model_dump(mode="json")
    r = client.post("/api/backtests", json=_body(strategy=s))
    assert r.status_code == 422
    assert "walk-forward" in r.json()["detail"].lower()


def test_backtest_walk_forward_holdout_counts(client):
    s = rsi_strategy(buy_level={"min": 25, "max": 35, "step": 5}).model_dump(mode="json")
    body = _body(strategy=s, mode="walk_forward", score_holdout=True)
    first = client.post("/api/backtests", json=body).json()["runs"][0]
    assert set(first["segments"]) == {"out_of_sample", "holdout"}
    assert first["segments"]["holdout"]["start"] >= "2021-09-01"
    assert first["segments"]["out_of_sample"]["end"] < "2021-09-01"
    assert first["holdout_count"] == 1
    assert len(first["windows"]) >= 2
    second = client.post("/api/backtests", json=body).json()["runs"][0]
    assert second["holdout_count"] == 2
    no_score = client.post("/api/backtests", json={**body, "score_holdout": False}).json()
    assert set(no_score["runs"][0]["segments"]) == {"out_of_sample"}
    assert no_score["runs"][0]["holdout_count"] == 2


def test_walk_forward_short_history_422(client):
    client.app_ref.dependency_overrides[get_price_provider] = lambda: ShortProvider()
    r = client.post("/api/backtests", json=_body(mode="walk_forward"))
    assert r.status_code == 422
    assert "AAPL" in r.json()["detail"] and "3.5 years" in r.json()["detail"]
    assert client.post("/api/backtests", json=_body()).status_code == 200


def test_never_trading_rule_serializes(client):
    s = rsi_strategy(buy_level=0.5)
    r = client.post("/api/backtests", json=_body(strategy=s.model_dump(mode="json")))
    assert r.status_code == 200
    data = json.loads(r.text)  # strict parse: NaN/Infinity would break this
    m = data["runs"][0]["segments"]["backtest"]["metrics"]
    assert m["num_trades"] == 0 and m["profit_factor"] is None


def test_multi_symbol_summary_median(client):
    r = client.post("/api/backtests", json=_body(symbols=["AAPL", "MSFT"])).json()
    assert [p["symbol"] for p in r["summary"]["per_symbol"]] == ["AAPL", "MSFT"]
    a, b = (p["metrics"]["total_return"] for p in r["summary"]["per_symbol"])
    assert r["summary"]["median"]["total_return"] == pytest.approx((a + b) / 2)


def test_settings_roundtrip(client):
    assert client.get("/api/settings").json() == {"holdout_start": "2021-09-01"}


class DownProvider:
    def get_daily_bars(self, symbol, start=None, end=None):
        raise ConnectionError("network down")


def test_data_source_down_503(client):
    client.app_ref.dependency_overrides[get_price_provider] = lambda: DownProvider()
    r = client.get("/api/bars/AAPL")
    assert r.status_code == 503
    assert "price data" in r.json()["detail"].lower()


@pytest.mark.parametrize(
    "start,end",
    [("2030-01-01", None), ("2020-06-01", "2020-01-01"), (None, "1990-01-01")],
)
def test_simple_dates_outside_data_422(client, start, end):
    r = client.post("/api/backtests", json=_body(start=start, end=end))
    assert r.status_code == 422
    assert "AAPL" in r.json()["detail"] or "start" in r.json()["detail"].lower()


def test_too_much_tuning_work_422(client):
    s = rsi_strategy(
        buy_level={"min": 10, "max": 40, "step": 1}, sell_level={"min": 60, "max": 65, "step": 1}
    ).model_dump(mode="json")
    r = client.post(
        "/api/backtests",
        json=_body(strategy=s, mode="walk_forward", symbols=["AAPL", "MSFT", "SPY"]),
    )
    assert r.status_code == 422
    assert "combinations" in r.json()["detail"]


def test_grid_error_message_passed_through_in_simple_mode(client):
    s = rsi_strategy(buy_level={"min": 0, "max": 1e9, "step": 1e-9}).model_dump(mode="json")
    r = client.post("/api/backtests", json=_body(strategy=s))
    assert r.status_code == 422
    assert "200" in r.json()["detail"]


class SpyDownProvider(FixtureProvider):
    def get_daily_bars(self, symbol, start=None, end=None):
        from app.data.providers import DataSourceError

        if symbol.upper() == "SPY":
            raise DataSourceError("down")
        return super().get_daily_bars(symbol, start, end)


def test_benchmark_outage_still_runs(client):
    client.app_ref.dependency_overrides[get_price_provider] = lambda: SpyDownProvider()
    r = client.post("/api/backtests", json=_body())
    assert r.status_code == 200
    assert r.json()["runs"][0]["baselines"]["spy"] is None
