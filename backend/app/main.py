from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import backtests, market, settings, strategies
from app.data.providers import DataSourceError, SymbolNotFound
from app.db import init_db

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Beat The Market", lifespan=lifespan)

    @app.exception_handler(SymbolNotFound)
    async def _not_found(request: Request, exc: SymbolNotFound):
        return JSONResponse({"detail": f"Unknown symbol: {exc.symbol}"}, status_code=404)

    async def _source_down(request: Request, exc: Exception):
        return JSONResponse(
            {
                "detail": "Couldn't reach the price data source. Check your internet connection "
                "and try again."
            },
            status_code=503,
        )

    for exc_type in (DataSourceError, ConnectionError, TimeoutError):
        app.add_exception_handler(exc_type, _source_down)

    @app.exception_handler(ValueError)
    async def _bad_value(request: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    for module in (market, strategies, backtests, settings):
        app.include_router(module.router)

    if FRONTEND_DIST.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            candidate = FRONTEND_DIST / path
            if path and candidate.is_file() and FRONTEND_DIST in candidate.resolve().parents:
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app


app = create_app()
