from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from app import repo
from app.db import get_session
from app.strategy.describe import describe
from app.strategy.models import Strategy

router = APIRouter(prefix="/api/strategies")


@router.post("/describe")
def describe_strategy(strategy: Strategy) -> dict:
    return {"text": describe(strategy)}


@router.get("")
def list_strategies(s: Session = Depends(get_session)) -> list[dict]:
    return repo.list_strategies(s)


@router.post("")
def create_strategy(strategy: Strategy, s: Session = Depends(get_session)) -> dict:
    return {"id": repo.save_strategy(s, strategy)}


@router.get("/{strategy_id}")
def get_strategy(strategy_id: int, s: Session = Depends(get_session)) -> dict:
    strategy = repo.get_strategy(s, strategy_id)
    if strategy is None:
        raise HTTPException(404, "Strategy not found")
    return {"id": strategy_id, "strategy": strategy.model_dump(mode="json")}


@router.delete("/{strategy_id}", status_code=204)
def delete_strategy(strategy_id: int, s: Session = Depends(get_session)) -> Response:
    if not repo.delete_strategy(s, strategy_id):
        raise HTTPException(404, "Strategy not found")
    return Response(status_code=204)
