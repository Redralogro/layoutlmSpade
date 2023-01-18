from datetime import datetime
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from helpers import infer, get_strings
from loss import BboxLoss
from modeling.warped_model import LitLayoutParsing
from pytorch_lightning import LightningModule
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import AutoTokenizer

print = tqdm.write

tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")


class LitBaseParsing(LightningModule):
    def __init__(self):
        super(LitBaseParsing, self).__init__()
        self.model = LitLayoutParsing()
        self.lr = 3e-5
        self.loss_clss = nn.CrossEntropyLoss()
        self.bbox_loss_fn = BboxLoss()

    @lru_cache
    def extend_matrix(self, matrix):
        matrix = matrix.cpu().numpy()
        matrix_s = [[0] + list(x) + [0] for x in list(matrix)]
        t_m = list(np.zeros_like(matrix_s[0]))
        _s = [t_m] + list(matrix_s) + [t_m]

        return np.array(_s, dtype='int8')

    @lru_cache
    def extend_label(self, label_):
        label_ = label_.cpu().numpy()
        label = [[0] + list(x) + [0] for x in list(label_)]
        return np.array(label, dtype='int')

    def forward(self, x: tuple) -> None:
        input_ids, attention_mask, token_type_ids, bbox, maps = x
        S, G = self.model((input_ids, attention_mask,
                          token_type_ids, bbox, maps))
        return S, G

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        def lf(x): return (1 - x / 81) * (1.0 - 0.1) + 0.1  # linear

        return {'optimizer': optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=lf),
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "metric_to_track"
                }
                }

    def training_step(self, batch, batch_idx) -> None:
        bbox = batch["bbox"].squeeze(0)
        maps = batch['maps']
        input_ids = batch['input_ids'].squeeze(0)
        attention_mask = batch['attention_mask'].squeeze(0)
        token_type_ids = batch['token_type_ids'].squeeze(0)
        # normalized_word_boxes = batch['normalized_word_boxes'].squeeze(0)
        ex_bboxes = bbox.squeeze(0)/1000
        S, G = self.forward(
            (input_ids, attention_mask, token_type_ids, bbox, maps)
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

        # print('loss', loss.detach())
        # print('loss_label_s', loss_label_s.detach())
        # print('loss_label_g', loss_label_g.detach())
        # print('loss_matrix_s', loss_matrix_s.detach())
        # print('loss_matrix_g', loss_matrix_g.detach())
        # print('loss_bboxes', bbox_loss.detach())
        self.log('Train/loss', loss.detach())
        self.log('Train/loss_bboxes', bbox_loss.detach())
        self.log('Train/loss_label_s', loss_label_s.detach())
        self.log('Train/loss_label_g', loss_label_g.detach())
        self.log('Train/loss_matrix_s', loss_matrix_s.detach())
        self.log('Train/loss_matrix_g', loss_matrix_g.detach())

        return loss

    def on_train_epoch_start(self) -> None:
        print('#########[TRAINING]###################\n')
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        print('#########[VALIDATING]###################\n')
        return super().on_validation_epoch_start()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> None:
        if (self.current_epoch % 2 == 0):

            bbox = batch["bbox"].squeeze(0)
            maps = batch['maps']
            input_ids = batch['input_ids'].squeeze(0)
            attention_mask = batch['attention_mask'].squeeze(0)
            token_type_ids = batch['token_type_ids'].squeeze(0)
            # normalized_word_boxes = batch['normalized_word_boxes'].squeeze(0)
            S, G = self.forward(
                (input_ids, attention_mask, token_type_ids, bbox, maps)
            )

            graph = torch.tensor(batch['label']).cuda()

            # GROUND TRUTH
            label_s = torch.tensor(self.extend_label(
                graph[0, :3, :])).unsqueeze(0).cuda()
            label_actual = label_s.squeeze(0)
            S_ = self.extend_matrix(graph[0, 3:, :])
            question_heads = [i for i, ele in enumerate(
                label_actual[0]) if ele != 0]
            answer_heads = [i for i, ele in enumerate(
                label_actual[1]) if ele != 0]
            # header_heads = [i for i, ele in enumerate(label_actual[2]) if ele != 0]
            text = [tokenizer.cls_token] + [x[0]
                                            for x in batch["text"]] + [tokenizer.sep_token]
            ques = get_strings(question_heads, text, S_)
            ans = get_strings(answer_heads, text, S_)
            print(f'[GROUND TRUTH]: Ques:{ques} \n Ans: {ans}')

            print('\n ###############################')
            # # PREDICT
            infer(S, G, text)

        return 0

    def validation_epoch_end(self, validation_step_outputs) -> None:
        if (self.current_epoch > 0 and self.current_epoch % 40 == 0):
            now = datetime.now()
            now = now.strftime("%d-%m-%Y_%H-%M-%S")
            print('Export .pt')
            with open(f"./resources/checkpoints/DP_model_finetune_{now}.pt", "wb") as f:
                torch.save(self.state_dict(), f)
        return 0
