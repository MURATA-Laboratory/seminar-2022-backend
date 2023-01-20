
from typing import Dict

from fastapi import Depends, FastAPI
from pydantic import BaseModel

import CaboCha as cb

import time

import numpy as np
import pytorch_lightning as pl
import torch
from box import Box
from transformers import BertModel, BertTokenizer

MODEL_PATH = "./epoch=3.ckpt"


app = FastAPI()


class IndentionRequest(BaseModel):
    text: str

    
class IndentionResponse(BaseModel):
    transla: str

  
config = dict(
    pretrained_model_name="cl-tohoku/bert-base-japanese-whole-word-masking",
    data_module=dict(
        batch_size=16,
        max_length=32,
    ),
    model=dict(
        hidden_lf_layer=256,
        hidden_comma_period_layer=2048,
    ),
)

config = Box(config)

tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name)
tokenizer.add_tokens(["[ANS]"])


class MyModel(pl.LightningModule):
    THRESHOLD = 0.5

    def __init__(
        self,
        tokenizer,
        pretrained_model_name,
        config,
    ):
        super().__init__()
        self.config = config

        self.bert = BertModel.from_pretrained(pretrained_model_name, return_dict=True)
        self.bert.resize_token_embeddings(len(tokenizer))

        # ラインフィードの判定 二値分類
        self.hidden_lf_layer = torch.nn.Linear(
            self.bert.config.hidden_size, config.model.hidden_lf_layer
        )
        self.lf_layer = torch.nn.Linear(config.model.hidden_lf_layer, 1)

        # 挿入なし, comma, periodの判定 三値分類
        self.hidden_comma_period_layer = torch.nn.Linear(
            self.bert.config.hidden_size, config.model.hidden_comma_period_layer
        )
        self.comma_period_layer = torch.nn.Linear(
            config.model.hidden_comma_period_layer, 3
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        lf_outputs = torch.relu(self.hidden_lf_layer(outputs.pooler_output))
        lf_predictions = torch.sigmoid(self.lf_layer(lf_outputs)).flatten()

        comma_period_outputs = torch.relu(
            self.hidden_comma_period_layer(outputs.pooler_output)
        )
        comma_period_predictions = torch.softmax(
            self.comma_period_layer(comma_period_outputs), dim=1  # row
        )
        return 0, [lf_predictions, comma_period_predictions]


model = MyModel(
    tokenizer,
    pretrained_model_name=config.pretrained_model_name,
    config=config,
)




model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"))["state_dict"]
)
model.eval()
model.freeze()

threshold = 0.5


@app.post("/predict", response_model=IndentionResponse)
def predict(request: IndentionRequest):
    # requestをcabochaを使って文節に区切る
    Intoken = ''
    translation = '' 
    cabocha = cb.Parser()
    tree = cabocha.parse(request)
    for i in range(tree.chunk_size()):
        chunk = tree.chunk(i)
        # 文節に区切る
        for ix in range(chunk.token_pos, chunk.token_pos + chunk.token_size):
            Intoken = Intoken + tree.token(ix).surface 
        # tokenを挿入
        if i < tree.chunk_size()-1:
            Intoken = Intoken + "[ANS]"

    # モデルとのやり取り
    encoding = tokenizer.encode_plus(
        Intoken,
        add_special_tokens=True,
        max_length=config.data_module.max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    predictions = model(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
    )[1]
    for i in range(tree.chunk_size()):
        # わからん translation = translation + Intoken.split("[ANS]")[i]
        if np.argmax(predictions[i+1]) == 1:
            translation = translation + "、"
        elif np.argmax(predictions[i+1]) == 2:
            translation = translation + "。"
        if predictions[0] > threshold:
            translation = translation
        # わからん translation = translation + Intoken.split("[ANS]")[i+1]
    return IndentionResponse(
        translation=translation
    )
