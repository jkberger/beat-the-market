from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import repo
from app.db import Base
from tests.strategy.helpers import rsi_strategy


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with sessionmaker(engine)() as s:
        yield s


def test_holdout_start_initializes_once(session):
    first = repo.get_holdout_start(session)
    assert abs((date.today() - first) - timedelta(days=730)) <= timedelta(days=2)
    assert repo.get_holdout_start(session) == first


def test_set_holdout_start(session):
    repo.set_holdout_start(session, date(2023, 1, 3))
    assert repo.get_holdout_start(session) == date(2023, 1, 3)


def test_record_holdout_score_increments(session):
    assert repo.holdout_count(session, "k", "AAPL") == 0
    assert repo.record_holdout_score(session, "k", "AAPL") == 1
    assert repo.record_holdout_score(session, "k", "AAPL") == 2
    assert repo.holdout_count(session, "k", "MSFT") == 0


def test_strategy_crud_roundtrip(session):
    s = rsi_strategy()
    sid = repo.save_strategy(session, s)
    assert repo.get_strategy(session, sid) == s
    assert [x["id"] for x in repo.list_strategies(session)] == [sid]
    repo.delete_strategy(session, sid)
    assert repo.get_strategy(session, sid) is None


def test_runs_listed_newest_first(session):
    a = repo.save_run(session, {"symbol": "AAPL", "mode": "simple", "strategy_name": "a"})
    b = repo.save_run(session, {"symbol": "MSFT", "mode": "simple", "strategy_name": "b"})
    assert [r["id"] for r in repo.list_runs(session)] == [b, a]
    assert repo.get_run(session, a)["symbol"] == "AAPL"
    assert repo.get_run(session, 999) is None
