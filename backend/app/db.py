from collections.abc import Iterator
from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import JSON, DateTime, Integer, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(255))


class SavedStrategy(Base):
    __tablename__ = "strategies"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200))
    body: Mapped[dict] = mapped_column(JSON)
    structure_key: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(200))
    mode: Mapped[str] = mapped_column(String(32))
    strategy_name: Mapped[str] = mapped_column(String(200))
    body: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class HoldoutScore(Base):
    __tablename__ = "holdout_scores"
    structure_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)


@lru_cache
def get_engine() -> Engine:
    url = get_settings().database_url
    if url.startswith("sqlite:///"):
        from pathlib import Path

        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True)


def init_db() -> None:
    Base.metadata.create_all(get_engine())


def get_session() -> Iterator[Session]:
    with sessionmaker(get_engine())() as session:
        yield session
