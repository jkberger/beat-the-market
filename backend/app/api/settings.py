from datetime import date

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import repo
from app.db import get_session

router = APIRouter(prefix="/api/settings")


class SettingsBody(BaseModel):
    holdout_start: date


@router.get("")
def get_settings(s: Session = Depends(get_session)) -> dict:
    return {"holdout_start": repo.get_holdout_start(s).isoformat()}


@router.put("")
def put_settings(body: SettingsBody, s: Session = Depends(get_session)) -> dict:
    repo.set_holdout_start(s, body.holdout_start)
    return {"holdout_start": body.holdout_start.isoformat()}
