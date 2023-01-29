import numpy as np
import pytorch_lightning as pl
import torch
from box import Box
from transformers import BertModel, BertTokenizer

LF_PERIOD_COMMA_MODEL_PATH = "./lf_comma_period_model.ckpt"

config = dict(
    pretrained_model_name="cl-tohoku/bert-base-japanese-whole-word-masking",
    data_module=dict(
        batch_size=16,
        max_length=32,
    ),
    model=dict(
        hidden_lf_layer=256,
        hidden_comma_period_layer=128,
    ),
)
config = Box(config)

tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name)
tokenizer.add_tokens(["[ANS]"])


class LfPeriodCommaModel(pl.LightningModule):
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
        return lf_predictions, comma_period_predictions

    def predict(self, text):
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=config.data_module.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return self(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )


model = LfPeriodCommaModel(
    tokenizer,
    pretrained_model_name=config.pretrained_model_name,
    config=config,
)
model.load_state_dict(
    torch.load(LF_PERIOD_COMMA_MODEL_PATH, map_location=torch.device("cpu"))[
        "state_dict"
    ]
)
model.eval()
model.freeze()


def get_model():
    return model
