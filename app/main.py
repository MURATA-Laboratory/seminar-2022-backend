import sys
import time

import CaboCha as cb
import numpy as np
from fastapi import Depends, FastAPI
from pydantic import BaseModel

sys.path.append("../")
from model import MyModel, get_model

app = FastAPI()

##### health check #####
@app.get("/health")
async def get_health():
    return {"status": "OK"}


##### predict #####
class InsertionRequest(BaseModel):
    text: str
    # TODO: last_chunkを受け取る処理を追加


class InsertionResponse(BaseModel):
    text: str
    last_chunk: str


@app.post("/predict", response_model=InsertionResponse)
def predict(request: InsertionRequest, model: MyModel = Depends(get_model)):
    cp = cb.Parser()
    tree = cp.parse(request.text)

    response = ""
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
        if lf > MyModel.THRESHOLD:
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
    return InsertionResponse(text=response, last_chunk=last_chunk_text)
