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
from helpers import infer
import json
from datetime import datetime

class LitLayoutParsing(LightningModule):
    def __init__(self):
        super(LitLayoutParsing, self).__init__()
        self.model = LayoutLMModel.from_pretrained(
            "microsoft/layoutlm-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/layoutlm-base-uncased")
        self.lr = 1e-4
        self.reduce_size = 256
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
        device = last_hidden_state.device
        reduce = []
        for g_token in maps:
            ten = self.zeros_temp.to(device)
            for ele in g_token:
                # print(ele)
                ten += last_hidden_state[0][i]
                # i+=1
                # print(last_hidden_state[0][i])
                i += 1
            ten = ten/len(g_token)
            # print(ten)
            reduce.append(ten)
        reduce = torch.stack(reduce).to(device)
        return reduce

    def forward(self, x: tuple) -> None:
        words, normalized_word_boxes = x
        device = normalized_word_boxes.device
        encoding = self.tokenizer(" ".join(words), return_tensors='pt')
        token_boxes = []
        i =0
        maps = []
        maps.extend([[i]])

        for word, box in zip(words, normalized_word_boxes):
            word_tokens = self.tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))
            i += 1
            maps.extend([[i]*len(word_tokens)])

        maps.extend([[i+1]])

        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        bbox = torch.tensor([token_boxes]).to(device)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        token_type_ids = encoding["token_type_ids"].to(device)
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
        bbox = batch["bbox"].squeeze(0)
        # maps = batch['maps']
        normalized_word_boxes = batch['normalized_word_boxes'].squeeze(0)
        ex_bboxes = bbox.squeeze(0)/1000
        S, G = self.forward(
            ([x[0]for x in batch["text"]], normalized_word_boxes)
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
        with torch.no_grad():
            text = [self.tokenizer.cls_token] + [x[0] for x in batch["text"]] + [self.tokenizer.sep_token]
            print('#########[TRAINING]###################\n')
            infer(S,G,text)
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
        print('loss_bboxes', bbox_loss.detach())

        self.log('Train/loss', loss.detach())
        self.log('Train/loss_bboxes', bbox_loss.detach())
        self.log('Train/loss_label_s', loss_label_s.detach())
        self.log('Train/loss_label_g', loss_label_g.detach())
        self.log('Train/loss_matrix_s', loss_matrix_s.detach())
        self.log('Train/loss_matrix_g', loss_matrix_g.detach())

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> None:
        print('#########[VALIDATING]###################\n')
        # bbox = batch["bbox"].squeeze(0)
        # maps = batch['maps']
        normalized_word_boxes = batch['normalized_word_boxes'].squeeze(0)
        # S, G = self.forward(
        #     (input_ids, bbox, attention_mask, token_type_ids, maps)
        # )

        S, G = self.forward(
            ([x[0]for x in batch["text"]], normalized_word_boxes)
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
        question_heads = [i for i, ele in enumerate(label_actual[0]) if ele != 0]
        answer_heads = [i for i, ele in enumerate(label_actual[1]) if ele != 0]
        # header_heads = [i for i, ele in enumerate(label_actual[2]) if ele != 0]
        text = [self.tokenizer.cls_token] + [x[0]
                                             for x in batch["text"]] + [self.tokenizer.sep_token]
        ques = get_strings(question_heads, text, S_)
        ans = get_strings(answer_heads, text, S_)
        print(f'[GROUND TRUTH]: Ques:{ques} \n Ans: {ans}')

        print('\n ###############################')
        # # PREDICT
        infer(S,G,text)
        print('\n ###############################')
        print('\n ###############################')
        print('\n ###############################')
        print('\n ###############################')
        print('\n ###############################')
        t_st = json.load(open('./data/processed/testing/1.jsonl'))


        def get_bbox(jsonl_file):
            w = jsonl_file['img_sz']['width']
            h = jsonl_file['img_sz']['height']
            bboxes = [[ int(x[0][0]*1000/w),int(x[0][1]*1000/h) ,int(x[2][0]*1000/w),int(x[2][1]*1000/h)] for x in jsonl_file['coord']]
            return  torch.tensor(bboxes)

        normed_bbox = get_bbox(t_st).cuda()
        words = t_st['text']
        
        S,G  = self.forward((words,normed_bbox))
        text = [self.tokenizer.cls_token] + words + [self.tokenizer.sep_token]
        infer(S,G,text)

        return 0

    def validation_epoch_end(self,validation_step_outputs) -> None:
        if (self.current_epoch > 0 and self.current_epoch % 150 == 0):
            now = datetime.now()
            now = now.strftime("%d-%m-%Y_%H-%M-%S")
            print('Export .pt')
            with open(f"./resources/checkpoints/DP_model_{now}.pt", "wb") as f:
                torch.save(self.state_dict(), f)
        return 0