import json
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.data.cache import CachedPriceProvider
from app.data.providers import PriceProvider, YahooProvider

UNIVERSE_FILE = Path(__file__).parent / "universe.json"


def load_universe() -> dict:
    return json.loads(UNIVERSE_FILE.read_text())


@lru_cache
def get_price_provider() -> PriceProvider:
    settings = get_settings()
    if settings.price_provider != "yahoo":
        raise ValueError(f"Unsupported price provider: {settings.price_provider}")
    return CachedPriceProvider(YahooProvider(), settings.data_dir / "prices")
