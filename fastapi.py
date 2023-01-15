
from typing import Dict
from fastapi import Depends, FastAPI
from pydantic import BaseModel
app = FastAPI()
class IndentionRequest(BaseModel):
    text: str
class IndentionResponse(BaseModel):
    caption: str
@app.post("/predict", response_model=IndentionResponse)
def predict(request: IndentionRequest):
    return IndentionResponse(
        caption=
    )