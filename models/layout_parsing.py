from tqdm import tqdm
from models.base import LitLayoutParsing
from models.loss import BboxLoss
from pytorch_lightning import LightningModule
import torch.nn as nn
from torch import Tensor
<<<<<<< HEAD


print = tqdm.write
=======
import numpy as np
from helpers import infer, get_strings
from functools import lru_cache
from torch import optim
from torch.optim import lr_scheduler
import torch
from datetime import datetime
from transformers import AutoTokenizer
print = tqdm.write

tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")


@lru_cache
def extend_matrix(matrix:Tensor):
    matrix = matrix.cpu().numpy()
    matrix_s = [[0] + list(x) + [0] for x in list(matrix)]
    t_m = list(np.zeros_like(matrix_s[0]))
    _s = [t_m] + list(matrix_s) + [t_m]

    return np.array(_s, dtype='int8')

@lru_cache
def extend_label(label_: Tensor):
    label_ = label_.cpu().numpy()
    label = [[0] + list(x) + [0] for x in list(label_)]
    return np.array(label, dtype='int')
>>>>>>> 9f1dd023b8eacd9eb6316eed4264a880142005c9

class LitBaseParsing(LightningModule):
    def __init__(self):
        super(LitBaseParsing, self).__init__()
        self.model = LitLayoutParsing()
<<<<<<< HEAD
        
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
=======
        self.loss_clss = nn.CrossEntropyLoss()
        self.bbox_loss_fn = BboxLoss()
        
    def forward(self, input_ids: Tensor, attention_mask: Tensor,
        token_type_ids:Tensor, bbox:Tensor, maps:Tensor) -> Tensor:
        S_G_graph = self.model(input_ids, attention_mask,
                          token_type_ids, bbox, maps)
        return S_G_graph
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-5)
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
        S_G_graph = self.forward(
            input_ids, attention_mask, token_type_ids, bbox, maps
        )
        S = S_G_graph[0].unsqueeze(0)
        G = S_G_graph[1].unsqueeze(0)
        
        s0, s1 = S[:, :, :3, :], S[:, :, 3:, :]
        g0, g1 = G[:, :, :3, :], G[:, :, 3:, :]
        graph = torch.tensor(batch['label']).cuda()
        
        S_ = extend_matrix(graph[0, 3:, :])
        G_ = extend_matrix(graph[1, 3:, :])
        # GROUND TRUTH
        label_s = torch.tensor(extend_label(
            graph[0, :3, :])).unsqueeze(0).cuda()
        label_g = torch.tensor(extend_label(
            graph[1, :3, :])).unsqueeze(0).cuda()
        matrix_s = torch.tensor(extend_matrix(
            graph[0, 3:, :])).unsqueeze(0).cuda()
        matrix_g = torch.tensor(extend_matrix(
            graph[1, 3:, :])).unsqueeze(0).cuda()
        label_actual = label_s.squeeze(0)
        
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
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> None:
        if (self.current_epoch % 2 == 0):

            bbox = batch["bbox"].squeeze(0)
            maps = batch['maps']
            input_ids = batch['input_ids'].squeeze(0)
            attention_mask = batch['attention_mask'].squeeze(0)
            token_type_ids = batch['token_type_ids'].squeeze(0)
            ex_bboxes = bbox.squeeze(0)/1000
            # normalized_word_boxes = batch['normalized_word_boxes'].squeeze(0)
            S_G_graph = self.forward(
            input_ids, attention_mask, token_type_ids, bbox, maps
            )
            S = S_G_graph[0].unsqueeze(0)
            G = S_G_graph[1].unsqueeze(0)

            s0, s1 = S[:, :, :3, :], S[:, :, 3:, :]
            g0, g1 = G[:, :, :3, :], G[:, :, 3:, :]
            graph = torch.tensor(batch['label']).cuda()

            # GROUND TRUTH
            label_s = torch.tensor(extend_label(
                graph[0, :3, :])).unsqueeze(0).cuda()
            label_g = torch.tensor(extend_label(
                graph[1, :3, :])).unsqueeze(0).cuda()
            matrix_s = torch.tensor(extend_matrix(
                graph[0, 3:, :])).unsqueeze(0).cuda()
            matrix_g = torch.tensor(extend_matrix(
                graph[1, 3:, :])).unsqueeze(0).cuda()
        
            label_actual = label_s.squeeze(0)
            S_ = extend_matrix(graph[0, 3:, :])
            G_ = extend_matrix(graph[1, 3:, :])
            question_heads = [i for i, ele in enumerate(label_actual[0]) if ele != 0]
            answer_heads = [i for i, ele in enumerate(label_actual[1]) if ele != 0]
            header_heads = [i for i, ele in enumerate(label_actual[2]) if ele != 0]
            text = [tokenizer.cls_token] + [x[0]
                                            for x in batch["text"]] + [tokenizer.sep_token]
            
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
            
            self.log('Val/loss', loss.detach())
            self.log('Val/loss_bboxes', bbox_loss.detach())
            self.log('Val/loss_label_s', loss_label_s.detach())
            self.log('Val/loss_label_g', loss_label_g.detach())
            self.log('Val/loss_matrix_s', loss_matrix_s.detach())
            self.log('Val/loss_matrix_g', loss_matrix_g.detach())
            
            
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
>>>>>>> 9f1dd023b8eacd9eb6316eed4264a880142005c9
