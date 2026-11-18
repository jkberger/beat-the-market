"""Record a fixture of daily bars from Yahoo for offline tests.

Usage: uv run python scripts/record_fixture.py AAPL 2014-01-01 2024-01-01 tests/fixtures/aapl_daily.csv
"""

import sys

import yfinance as yf


def main() -> None:
    symbol, start, end, out = sys.argv[1:5]
    df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
    df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.round(6).to_csv(out)
    print(f"wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
