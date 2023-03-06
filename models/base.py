import torch.nn as nn
import torch
from pytorch_lightning import LightningModule
from functools import lru_cache
from transformers import LayoutLMModel
from torch import Tensor
import numpy as np

@lru_cache
def load_layoutModel():
    return LayoutLMModel.from_pretrained(
            "microsoft/layoutlm-base-uncased", local_files_only=True)


class RelationTagger(nn.Module):
    def __init__(self, n_fields, hidden_size, head_p_dropout=0.1):
        super().__init__()
        self.head = nn.Linear(hidden_size, hidden_size)
        self.tail = nn.Linear(hidden_size, hidden_size)
        self.field_embeddings = nn.Parameter(
            torch.rand(1, n_fields, hidden_size))
        self.W_label_0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_label_1 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, enc):

        enc_head = self.head(enc)
        enc_tail = self.tail(enc)

        batch_size = enc_tail.size(0)
        field_embeddings = self.field_embeddings.expand(batch_size, -1, -1)
        enc_head = torch.cat([field_embeddings, enc_head], dim=1)

        score_0 = torch.matmul(
            enc_head, self.W_label_0(enc_tail).transpose(1, 2))
        score_1 = torch.matmul(
            enc_head, self.W_label_1(enc_tail).transpose(1, 2))

        score = torch.cat(
            [
                score_0.unsqueeze(1),
                score_1.unsqueeze(1),
            ],
            dim=1,
        )
        return score

class RelativePositionAttention(nn.Module):
    def __init__(self, embed_size, num_attention_heads=1):
        super().__init__()
        self.Q = nn.Linear(embed_size, embed_size *
                           num_attention_heads, bias=False)
        self.K = nn.Linear(embed_size, embed_size *
                           num_attention_heads, bias=False)
        self.V = nn.Linear(embed_size, embed_size *
                           num_attention_heads, bias=False)
        self.num_attention_heads = num_attention_heads
        self.project = nn.Linear(
            embed_size * num_attention_heads, embed_size, bias=False)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, position_embeddings, values):
        position_embeddings = self.norm(position_embeddings)

        # Input: batch * time * hidden
        Q = self.Q(position_embeddings)
        K = self.K(position_embeddings)
        V = self.V(values)

        # [b, t, 1, d] and [b, 1, t, d] broadcast
        W = Q.unsqueeze(-2) - K.unsqueeze(-3)
        # [b, t, t, d, heads]
        W = torch.stack(W.chunk(self.num_attention_heads, dim=-1), dim=-1)
        W = torch.softmax(W, dim=-3)

        # [b, t, t, d, heads]
        V = torch.stack(V.chunk(self.num_attention_heads, -1), -1)

        # to [b, t, d]
        # print(W.shape, V.shape)
        outputs = torch.einsum("btdh,btwdh->btdh", V, W)
        outputs = self.project(outputs.flatten(-2))
        return outputs


class MyPositionEmbedding(nn.Module):
    def __init__(self, config, embed_size):
        super().__init__()
        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, embed_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, embed_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, embed_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, embed_size
        )

    def forward(self, bboxes: torch.Tensor):
        # boxes: [b, n, 4]
        x1, y1, x2, y2 = bboxes.chunk(4, dim=-1)
        w = x2 - x1
        h = y2 - y1
        embs = (self.x_position_embeddings(x1) +
                self.y_position_embeddings(y1) +
                self.w_position_embeddings(w) +
                self.h_position_embeddings(h))
        embs = embs.squeeze(-2)
        return embs



class LitLayoutParsing(LightningModule):
    def __init__(self):
        super(LitLayoutParsing, self).__init__()
        self.model = LayoutLMModel.from_pretrained(
            "microsoft/layoutlm-base-uncased", local_files_only=True)

        self.reduce_size = 256

        self.rel_attn = RelativePositionAttention(self.reduce_size, 8)
        self.pos_embs = MyPositionEmbedding(
            self.model.config, self.reduce_size)

        self.ln = nn.Linear(self.model.config.hidden_size, self.reduce_size)
        self.dropout = nn.Dropout(0.1)
        self.zeros_temp = torch.zeros(self.reduce_size)

        self.rel_s = RelationTagger(
            hidden_size=self.reduce_size,
            n_fields=3,
        )

        self.rel_g = RelationTagger(
            hidden_size=self.reduce_size,
            n_fields=3,
        )

        self.S: Tensor = None
        self.G: Tensor = None

    def load_config_param(self):
        self.model.config.num_attention_heads = 4
        self.model.config.hidden_size = self.reduce_size
        self.model.config.num_hidden_layers = 5

    def reduce_shape(self, last_hidden_state, maps):
        i = 0
        device = last_hidden_state.device
        reduce = []
        for g_token in maps:
            ten = self.zeros_temp.to(device)
            for ele in g_token:
                ten += last_hidden_state[0][i]
                i += 1
            ten = ten/len(g_token)
            reduce.append(ten)
        reduce = torch.stack(reduce).to(device)
        return reduce

    def tranfer_maps(self, maps: Tensor):
        maps = torch.squeeze(maps, 0)
        # maps = maps.detach()
        # m = [ item.item() -0 for item in maps]
        # box_index =  np.unique(m).tolist()
        m = np.array(maps.cpu())
        m = m.tolist()
        box_index =  np.unique(m).tolist()
        return [[item]*m.count(item) for item in box_index]

    def change_type(self,ten: Tensor):
        return ten.long()

    def forward(self, input_ids: Tensor, attention_mask: Tensor,
        token_type_ids:Tensor, bbox:Tensor, maps:Tensor) -> Tensor:
        device = self.device
        # input_ids, attention_mask, token_type_ids, bbox, maps = x
        input_ids = self.change_type(input_ids)
        attention_mask = self.change_type(attention_mask)
        token_type_ids = self.change_type(token_type_ids)
        bbox = self.change_type(bbox)
        maps = self.change_type(maps)

        outputs = self.model(input_ids=input_ids,bbox=bbox,attention_mask=attention_mask,token_type_ids=token_type_ids)
        maps = self.tranfer_maps(maps)
        # final feature
        last_hidden_state = self.ln(outputs.last_hidden_state)
        position_embeddings = self.pos_embs(bbox)
        last_hidden_state = self.rel_attn(
                                position_embeddings,
                                last_hidden_state
                            )
        reduce = self.reduce_shape(last_hidden_state, maps)

        # s part
        self.S = self.rel_s(self.dropout(reduce.unsqueeze(0)))
        # group part
        self.G = self.rel_g(self.dropout(reduce.unsqueeze(0)))

        return  torch.cat((self.S , self.G),0).to(device)
