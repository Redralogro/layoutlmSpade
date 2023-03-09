import json
from functools import lru_cache
from traceback import print_exc
from typing import Dict

import networkx as nx
# from modeling.warped_model import LitLayoutParsing
import numpy as np
import torch
import torch.nn as nn
import yaml



def get_strings(heads, data: list, graph): 
    temp = []
    # G = nx.Graph(graph_s[0,3:,:]) # s
    G = nx.Graph(graph) # s
    try:
        for index in heads:
            dfs = list(nx.dfs_edges(G, source=int(index)))
            if  dfs == []:
                header = [int(index)]
            else: header =  [dfs[0][0]] + [x[1]  for i,x in enumerate (dfs)]
            str_ = ''
            for i in header:
                str_ += ' ' + data[int(i)] 
                assert i <= len(data)
            temp.append([index, str_[1:]])
    except Exception:
        pass
    return temp

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


def b_loss(graph_matrix, graph_matrix_S, ex_bboxes, list_heads):
    question_heads, answer_heads, pred_question_heads, pred_answer_heads = list_heads

    def bbox_loss(graph, ex_bboxes, heads):
        temp = []
        G = nx.Graph(graph)  # s
        for index in heads:
            dfs = list(nx.dfs_edges(G, source=int(index)))
            dfs
            if dfs == []:
                header = [int(index)]
            else:
                header = [dfs[0][0]] + [x[1] for i, x in enumerate(dfs)]
            list_temp = []
            for i in header:
                [x1, y1, x2, y2] = ex_bboxes[int(i)]
                list_temp.append(abs(x2 - x1) + abs(y2 - y1))
            temp.append(sum(list_temp)/len(list_temp))
        try:
            return sum(temp)/len(temp)
        except ZeroDivisionError:
            return 0

    que_loss = bbox_loss(graph_matrix, ex_bboxes, question_heads)
    ans_loss = bbox_loss(graph_matrix, ex_bboxes, answer_heads)
    pred_ques_loss = bbox_loss(graph_matrix_S, ex_bboxes, pred_question_heads)
    pred_ans_loss = bbox_loss(graph_matrix_S, ex_bboxes, pred_answer_heads)
    sum_loss = abs(pred_ques_loss - que_loss) + abs(pred_ans_loss - ans_loss)
    return torch.tensor(sum_loss, requires_grad=True)


def tensorize(x):
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x)
    else:
        try:
            return torch.as_tensor(np.array(x))
        except Exception:
            print_exc()
            return torch.tensor(x)


def normalize_box(box, width, height):
    # x1, y1, x2, y2
    normed = [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]
    return np.clip(normed, 0, 1000)


def load_config(path: str) -> Dict:
    if path.endswith("yml"):
        with open(path, "r") as f:
            config = yaml.load(stream=f, Loader=yaml.FullLoader)
    elif path.endswith("json"):
        with open(path, "r") as f:
            config = json.load(stream=f)
    elif path.endswith("yaml"):
        with open(path, "r") as f:
            config = yaml.load(stream=f, Loader=yaml.FullLoader)
    else:
        raise "Invalid file format"
    return config

def infer(S,G,text):
    s0, s1 = S[:, :, :3, :], S[:, :, 3:, :]
    g0, g1 = G[:, :, :3, :], G[:, :, 3:, :]
    pred_matrix_s = torch.softmax(s1, dim=1)
    pred_matrix_s = torch.argmax(pred_matrix_s, dim=1).squeeze(0)
    pred_matrix_g = torch.softmax(g1, dim=1)
    pred_matrix_g = torch.argmax(pred_matrix_g, dim=1).squeeze(0)
    pred_label = torch.argmax(s0, dim=1).squeeze(0)
    pred_S = np.array([list(x) for x in np.array(pred_matrix_s.cpu().numpy())])
    pred_G = np.array([list(x) for x in np.array(pred_matrix_g.cpu().numpy())])
    pred_question_heads = [
        i for i, ele in enumerate(pred_label[0]) if ele != 0]
    pred_answer_heads = [
        i for i, ele in enumerate(pred_label[1]) if ele != 0]
    
    pred_ques = get_strings(pred_question_heads, text, pred_S)
        # print(np.shape(ques))

    pred_ans = get_strings(pred_answer_heads, text, pred_S)
    print(f'[PREDICT]: Ques:{pred_ques} \n Ans: {pred_ans}')
    print(f'[PREDICT MAPPING]')
    for ques_idx in pred_question_heads:
            G_pred = nx.Graph(pred_G)  # group
            dfs = list(nx.dfs_edges(G_pred, source=int(ques_idx)))
            # print(dfs)
            if len(dfs) != 0:
                q, a = dfs[0]
                qu_s = [qs[1] for qs in pred_ques if q in qs]
                an_s = [as_[1] for as_ in pred_ans if a in as_]
                # if len(qu_s)== len(an_s):
                #     print(qu_s[0], an_s[0])
                try:
                    print('============================================================')
                    print(f'{qu_s[0]}|{an_s[0]}')
                except Exception:
                    pass
