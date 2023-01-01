from pytorch_lightning import LightningModule
from PIL import Image
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, LayoutLMModel
import numpy as np
from torch import Tensor, optim
import torch
from spade_model import RelationTagger
from loss import BboxLoss
from graph_stuff import get_strings, get_qa
import networkx as nx


class LitLayoutParsing(LightningModule):
    def __init__(self):
        super(LitLayoutParsing, self).__init__()
        self.model = LayoutLMModel.from_pretrained(
            "microsoft/layoutlm-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/layoutlm-base-uncased")
        self.lr = 1e-4
        self.reduce_size = 256
        self.config = AutoConfig.from_pretrained(
            "microsoft/layoutlm-base-uncased")
        self.ln = nn.Linear(self.config.hidden_size, self.reduce_size)
        self.dropout = nn.Dropout(0.1)

        self.rel_s = RelationTagger(
            hidden_size=self.reduce_size,
            n_fields=3,
        )

        self.rel_g = RelationTagger(
            hidden_size=self.reduce_size,
            n_fields=3,
        )
        self.loss_clss = nn.CrossEntropyLoss()
        self.bbox_loss_fn = BboxLoss()

    def extend_matrix(self, matrix):
        matrix = matrix.cpu().numpy()
        matrix_s = [[0] + list(x) + [0] for x in list(matrix)]
        t_m = list(np.zeros_like(matrix_s[0]))
        _s = [t_m] + list(matrix_s) + [t_m]

        return np.array(_s, dtype='int8')

    def extend_label(self, label_):
        label_ = label_.cpu().numpy()
        label = [[0] + list(x) + [0] for x in list(label_)]
        return np.array(label, dtype='int')

    def reduce_shape(self, last_hidden_state, maps):
        i = 0
        # reduce = torch.zeros(768)
        reduce = []
        for g_token in maps:
            ten = torch.zeros(self.reduce_size).cuda()
            for ele in g_token:
                # print(ele)
                ten += last_hidden_state[0][i]
                # i+=1
                # print(last_hidden_state[0][i])
                i += 1
            ten = ten/len(g_token)
            # print(ten)
            reduce.append(ten)
            # reduce = torch.cat((reduce,ten),-1)
        # print(np.array(reduce).shape)
        # print(reduce)
        reduce = torch.stack(reduce)
        return reduce

    def forward(self, x: tuple) -> None:
        input_ids, bbox, attention_mask, token_type_ids, maps = x
        outputs = self.model(input_ids=input_ids, bbox=bbox,
                             attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = self.ln(outputs.last_hidden_state)
        reduce = self.reduce_shape(last_hidden_state, maps)
        # s part
        S = self.rel_s(self.dropout(reduce.unsqueeze(0)))
        # group part
        G = self.rel_g(self.dropout(reduce.unsqueeze(0)))

        return S, G

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx) -> None:
        input_ids = batch["input_ids"].squeeze(0)
        attention_mask = batch["attention_mask"].squeeze(0)
        token_type_ids = batch["token_type_ids"].squeeze(0)
        bbox = batch["bbox"].squeeze(0)
        maps = batch['maps']
        ex_bboxes = bbox.squeeze(0)/1000
        S, G = self.forward(
            (input_ids, bbox, attention_mask, token_type_ids, maps)
        )
        s0, s1 = S[:, :, :3, :], S[:, :, 3:, :]
        g0, g1 = G[:, :, :3, :], G[:, :, 3:, :]
        graph = torch.tensor(batch['label']).cuda()
        # GROUND TRUTH
        label_s = torch.tensor(self.extend_label(
            graph[0, :3, :])).unsqueeze(0).cuda()
        label_g = torch.tensor(self.extend_label(
            graph[1, :3, :])).unsqueeze(0).cuda()
        matrix_s = torch.tensor(self.extend_matrix(
            graph[0, 3:, :])).unsqueeze(0).cuda()
        matrix_g = torch.tensor(self.extend_matrix(
            graph[1, 3:, :])).unsqueeze(0).cuda()
        label_actual = label_s.squeeze(0)
        S_ = self.extend_matrix(graph[0, 3:, :])
        G_ = self.extend_matrix(graph[1, 3:, :])
        question_heads = [i for i, ele in enumerate(
            label_actual[0]) if ele != 0]
        answer_heads = [i for i, ele in enumerate(label_actual[1]) if ele != 0]
        header_heads = [i for i, ele in enumerate(label_actual[2]) if ele != 0]
        # PREDICT
        pred_matrix_s = torch.softmax(s1, dim=1)
        pred_matrix_s = torch.argmax(pred_matrix_s, dim=1).squeeze(0)
        pred_label = torch.argmax(s0, dim=1).squeeze(0)
        pred_S = np.array([list(x)
                          for x in np.array(pred_matrix_s.cpu().numpy())])
        # pred_G = np.array([list(x) for x in np.array(pred_matrix_g.cpu().numpy())])
        pred_question_heads = [
            i for i, ele in enumerate(pred_label[0]) if ele != 0]
        pred_answer_heads = [
            i for i, ele in enumerate(pred_label[1]) if ele != 0]
        bbox_loss = self.bbox_loss_fn(S_, pred_S, ex_bboxes, (question_heads,
                                                              answer_heads,
                                                              pred_question_heads,
                                                              pred_answer_heads))

        loss_label_s = self.loss_clss(s0, label_s.long())
        loss_label_g = self.loss_clss(g0, label_g.long())
        loss_matrix_s = self.loss_clss(s1, matrix_s.long())
        loss_matrix_g = self.loss_clss(g1, matrix_g.long())
        loss = loss_label_s + loss_matrix_s + loss_label_g + loss_matrix_g + bbox_loss

        print('loss', loss.detach())
        print('loss_label_s', loss_label_s.detach())
        print('loss_label_g', loss_label_g.detach())
        print('loss_matrix_s', loss_matrix_s.detach())
        print('loss_matrix_g', loss_matrix_g.detach())
        print('loss_bboxes', bbox_loss)
        self.log('Train/loss', loss.detach())
        self.log('Train/loss_bboxes', bbox_loss)
        self.log('Train/loss_label_s', loss_label_s.detach())
        self.log('Train/loss_label_g', loss_label_g.detach())
        self.log('Train/loss_matrix_s', loss_matrix_s.detach())
        self.log('Train/loss_matrix_g', loss_matrix_g.detach())

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> None:
        input_ids = batch["input_ids"].squeeze(0)
        attention_mask = batch["attention_mask"].squeeze(0)
        token_type_ids = batch["token_type_ids"].squeeze(0)
        bbox = batch["bbox"].squeeze(0)
        maps = batch['maps']
        S, G = self.forward(
            (input_ids, bbox, attention_mask, token_type_ids, maps)
        )

        s0, s1 = S[:, :, :3, :], S[:, :, 3:, :]
        graph = torch.tensor(batch['label']).cuda()

        g0, g1 = G[:, :, :3, :], G[:, :, 3:, :]
        # GROUND TRUTH
        label_s = torch.tensor(self.extend_label(
            graph[0, :3, :])).unsqueeze(0).cuda()
        label_g = torch.tensor(self.extend_label(
            graph[1, :3, :])).unsqueeze(0).cuda()
        matrix_s = torch.tensor(self.extend_matrix(
            graph[0, 3:, :])).unsqueeze(0).cuda()
        matrix_g = torch.tensor(self.extend_matrix(
            graph[1, 3:, :])).unsqueeze(0).cuda()
        label_actual = label_s.squeeze(0)
        S_ = self.extend_matrix(graph[0, 3:, :])
        G_ = self.extend_matrix(graph[1, 3:, :])
        question_heads = [i for i, ele in enumerate(
            label_actual[0]) if ele != 0]
        answer_heads = [i for i, ele in enumerate(label_actual[1]) if ele != 0]
        header_heads = [i for i, ele in enumerate(label_actual[2]) if ele != 0]
        text = [self.tokenizer.cls_token] + [x[0]
                                             for x in batch["text"]] + [self.tokenizer.sep_token]
        ques = get_strings(question_heads, text, S_)
        ans = get_strings(answer_heads, text, S_)
        print(f'[GROUND TRUTH]: Ques:{ques} \n Ans: {ans}')

        print('\n ###############################')
        # PREDICT
        pred_matrix_s = torch.softmax(s1, dim=1)
        pred_matrix_s = torch.argmax(pred_matrix_s, dim=1).squeeze(0)
        pred_matrix_g = torch.softmax(g1, dim=1)
        pred_matrix_g = torch.argmax(pred_matrix_g, dim=1).squeeze(0)
        pred_label = torch.argmax(s0, dim=1).squeeze(0)
        pred_S = np.array([list(x)
                          for x in np.array(pred_matrix_s.cpu().numpy())])
        pred_G = np.array([list(x)
                          for x in np.array(pred_matrix_g.cpu().numpy())])
        # pred_G = np.array([list(x) for x in np.array(pred_matrix_g.cpu().numpy())])
        pred_question_heads = [
            i for i, ele in enumerate(pred_label[0]) if ele != 0]
        pred_answer_heads = [
            i for i, ele in enumerate(pred_label[1]) if ele != 0]

        pred_ques = get_strings(pred_question_heads, text, pred_S)
        # print(np.shape(ques))

        pred_ans = get_strings(pred_answer_heads, text, pred_S)
        print(f'[PREDICT]: Ques:{pred_ques} \n Ans: {pred_ans}')

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
                print('============================================================')
                print(f'[PREDICT MAPPING]: ques: {qu_s} \n ans: {an_s}')

        return 0
