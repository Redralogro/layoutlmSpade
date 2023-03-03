from tqdm import tqdm
from models.base import LitLayoutParsing
from models.loss import BboxLoss
from pytorch_lightning import LightningModule
import torch.nn as nn
from torch import Tensor


print = tqdm.write

class LitBaseParsing(LightningModule):
    def __init__(self):
        super(LitBaseParsing, self).__init__()
        self.model = LitLayoutParsing()
        
    def forward(self, input_: Tensor,bbox_:Tensor, maps:Tensor) -> Tensor:
        input_ids = input_[0].unsqueeze(0)
        attention_mask = input_[1].unsqueeze(0)
        token_type_ids = input_[2].unsqueeze(0)
        maps = maps
        bbox = bbox_
        S_G_graph = self.model(maps, input_ids, attention_mask,
                                token_type_ids, bbox)
        return S_G_graph, maps
        # (input_ids ,bbox_, maps)
        # S_G_graph