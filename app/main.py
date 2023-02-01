import sys
import time
import os

import CaboCha as cb
import numpy as np
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


sys.path.append("../")
from model import LfPeriodCommaModel, get_model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # set string like "http://localhost:8000,https://example.com"
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##### health check #####
@app.get("/health")
async def get_health():
    return {"status": "OK"}


##### predict #####
class InsertionRequest(BaseModel):
    text: str
    pre_last_chunk: str | None = None


class InsertionResponse(BaseModel):
    text: str
    last_chunk: str
    elapsed_time: float


@app.post("/predict", response_model=InsertionResponse)
def predict(request: InsertionRequest, model: LfPeriodCommaModel = Depends(get_model)):
    start = time.time()

    cp = cb.Parser()
    tree = cp.parse(request.text)

    response = ""
    if request.pre_last_chunk is not None:
        chunk = tree.chunk(0)
        chunk_text = "".join(
            [
                tree.token(j).surface
                for j in range(chunk.token_pos, chunk.token_pos + chunk.token_size)
            ]
        )
        lf, comma_period = model.predict(request.pre_last_chunk + "[ANS]" + chunk_text)

        if np.argmax(comma_period) == 1:
            response += "、"
        elif np.argmax(comma_period) == 2:
            response += "。"
        if lf > LfPeriodCommaModel.THRESHOLD:
            response += "\n"

    for i in range(tree.chunk_size() - 1):
        chunk = tree.chunk(i)
        chunk_text = "".join(
            [
                tree.token(j).surface
                for j in range(chunk.token_pos, chunk.token_pos + chunk.token_size)
            ]
        )
        next_chunk = tree.chunk(i + 1)
        next_chunk_text = "".join(
            [
                tree.token(j).surface
                for j in range(
                    next_chunk.token_pos, next_chunk.token_pos + next_chunk.token_size
                )
            ]
        )
        lf, comma_period = model.predict(chunk_text + "[ANS]" + next_chunk_text)

        response += chunk_text
        if np.argmax(comma_period) == 1:
            response += "、"
        elif np.argmax(comma_period) == 2:
            response += "。"
        if lf > LfPeriodCommaModel.THRESHOLD:
            response += "\n"

    last_chunk_text = "".join(
        [
            tree.token(j).surface
            for j in range(
                tree.chunk(tree.chunk_size() - 1).token_pos,
                tree.chunk(tree.chunk_size() - 1).token_pos
                + tree.chunk(tree.chunk_size() - 1).token_size,
            )
        ]
    )
    response += last_chunk_text
    return InsertionResponse(
        text=response, last_chunk=last_chunk_text, elapsed_time=time.time() - start
    )
