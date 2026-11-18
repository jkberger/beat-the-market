import os
import time
from datetime import date

import pytest

from app.data.cache import CachedPriceProvider
from app.data.providers import SymbolNotFound
from tests.conftest import load_fixture_bars


class FakeProvider:
    def __init__(self):
        self.calls = 0
        self.fail: Exception | None = None

    def get_daily_bars(self, symbol, start=None, end=None):
        self.calls += 1
        if self.fail:
            raise self.fail
        if symbol == "NOPE":
            raise SymbolNotFound(symbol)
        return load_fixture_bars()


def _age(path, days):
    t = time.time() - days * 86400
    os.utime(path, (t, t))


def test_cache_uses_inner_once_within_max_age(tmp_path):
    inner = FakeProvider()
    p = CachedPriceProvider(inner, tmp_path)
    p.get_daily_bars("AAPL")
    p.get_daily_bars("AAPL")
    assert inner.calls == 1


def test_cache_refetches_when_stale(tmp_path):
    inner = FakeProvider()
    p = CachedPriceProvider(inner, tmp_path)
    p.get_daily_bars("AAPL")
    _age(tmp_path / "AAPL.parquet", 2)
    p.get_daily_bars("AAPL")
    assert inner.calls == 2


def test_cache_falls_back_when_inner_fails(tmp_path):
    inner = FakeProvider()
    p = CachedPriceProvider(inner, tmp_path)
    first = p.get_daily_bars("AAPL")
    _age(tmp_path / "AAPL.parquet", 2)
    inner.fail = ConnectionError("down")
    second = p.get_daily_bars("AAPL")
    assert len(second) == len(first)


def test_cache_propagates_symbol_not_found(tmp_path):
    p = CachedPriceProvider(FakeProvider(), tmp_path)
    with pytest.raises(SymbolNotFound):
        p.get_daily_bars("NOPE")


def test_slices_start_end(tmp_path):
    p = CachedPriceProvider(FakeProvider(), tmp_path)
    df = p.get_daily_bars("AAPL", start=date(2016, 1, 1), end=date(2017, 1, 1))
    assert df.index.min().date() >= date(2016, 1, 1)
    assert df.index.max().date() <= date(2017, 1, 1)
    assert len(df) > 200


def test_symbol_is_case_insensitive(tmp_path):
    inner = FakeProvider()
    p = CachedPriceProvider(inner, tmp_path)
    p.get_daily_bars("aapl")
    p.get_daily_bars("AAPL")
    assert inner.calls == 1


def test_cache_serves_stale_file_when_source_says_not_found(tmp_path):
    inner = FakeProvider()
    p = CachedPriceProvider(inner, tmp_path)
    first = p.get_daily_bars("AAPL")
    _age(tmp_path / "AAPL.parquet", 2)
    inner.fail = SymbolNotFound("AAPL")
    assert len(p.get_daily_bars("AAPL")) == len(first)
