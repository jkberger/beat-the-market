from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture_bars() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "synthetic_daily.csv", index_col="date", parse_dates=True)
    df["adj_close"] = df["close"]
    return df[["open", "high", "low", "close", "adj_close", "volume"]]


@pytest.fixture
def bars() -> pd.DataFrame:
    return load_fixture_bars()
