from datetime import datetime
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import AutoTokenizer

from infer.model.loss import BboxLoss
from infer.model.warped_model import LitLayoutParsing

print = tqdm.write

tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")


class LitBaseParsing(LightningModule):
    def __init__(self):
        super(LitBaseParsing, self).__init__()
        self.model = LitLayoutParsing()
        self.lr = 3e-5
        self.loss_clss = nn.CrossEntropyLoss()
        self.bbox_loss_fn = BboxLoss()


    def forward(self, x: tuple) -> None:
        input_ids, attention_mask, token_type_ids, bbox, maps = x
        S, G = self.model((input_ids, attention_mask,
                          token_type_ids, bbox, maps))
        return S, G

