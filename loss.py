import networkx as nx
import torch
import math
import torch.nn as nn
from gensim.models import Word2Vec

class BboxLoss(nn.Module):
    def __init__(self,  weight=None):
        super(BboxLoss, self).__init__()

    def forward(self, gp_, gp_S, ex_bboxes,text, list_heads):
        question_heads, answer_heads, pred_question_heads, pred_answer_heads = list_heads
        que_loss = self.bbox_loss(gp_, ex_bboxes, question_heads)
        ans_loss = self.bbox_loss(gp_, ex_bboxes, answer_heads)
        pred_ques_loss = self.bbox_loss(gp_S, ex_bboxes, pred_question_heads)
        pred_ans_loss = self.bbox_loss(gp_S, ex_bboxes, pred_answer_heads)

        summary_loss = torch.tensor(
            [abs(pred_ques_loss - que_loss), abs(pred_ans_loss - ans_loss)], requires_grad=True)
        return torch.sum(summary_loss)


    def wordemd_dict(self,text):
        model = Word2Vec(sentences=[[item] for item in text] , vector_size=100, window=5, min_count=1, workers=4,  sg= 1)
        
        return model.wv
        
        return text
    def bbox_loss(self, graph, ex_bboxes, heads):
        temp = []
        G = nx.Graph(graph)  # s
        for index in heads:
            dfs = list(nx.dfs_edges(G, source=int(index)))
            if dfs == []:
                header = [int(index)]
            else:
                header = [dfs[0][0]] + [x[1] for i, x in enumerate(dfs)]
            list_temp = []
            print(header)
            for i in header:
                [x1, y1, x2, y2] = ex_bboxes[int(i)]
                list_temp.append(math.sqrt(abs(x2 - x1)**2 + abs(y2 - y1)**2))
            temp.append(sum(list_temp)/len(list_temp))
        try:
            return sum(temp)/len(temp)
        except ZeroDivisionError:
            return 0
