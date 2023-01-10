from functools import cache
from typing import List

from fastapi import APIRouter, Depends, HTTPException

import api.cruds.user as user_crud

import api.schemas.mock as mock_schema

router = APIRouter()

@router.get("/pages", response_model=mock_schema.Page)
async def reverse(mock_id:int):
    return mock_schema.Page(request=str(mock_id))

@router.post("/pages", response_model=mock_schema.Page)
async def create(mock_body: user_schema.PageCreate):
    return mock_schema.PageCreat(id=1)
