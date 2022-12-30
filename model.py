from pytorch_lightning import LightningModule
from layoutlm import layoutlmBase
from PIL import Image
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, LayoutLMModel
from torch import Tensor, optim
import torch
from spade_model import RelationTagger
import numpy as np
from graph_stuff import get_strings, get_qa

class spadeLayoutLM(LightningModule):
    def __init__(self):
        super(spadeLayoutLM, self).__init__()
        self.model  = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
        self.lr = 1e-4
        self.reduce_size = 256
        self.config = AutoConfig.from_pretrained("microsoft/layoutlm-base-uncased")
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
        self.lss = nn.CrossEntropyLoss()
    
    def forward(self, x: None) -> None:
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def reduce_shape(self, last_hidden_state, maps):
        i = 0
        # reduce = torch.zeros(768)
        reduce =[]
        for g_token in  maps:
            ten = torch.zeros(self.reduce_size).cuda()
            for ele in g_token:
                # print(ele)
                ten += last_hidden_state[0][i]
                # i+=1
                # print(last_hidden_state[0][i])
                i+=1
            ten = ten/len(g_token)
            # print(ten)
            reduce.append(ten)
            # reduce = torch.cat((reduce,ten),-1)
        # print(np.array(reduce).shape)
        # print(reduce)
        reduce=  torch.stack(reduce)
        return reduce

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].squeeze(0)
        attention_mask = batch["attention_mask"].squeeze(0)
        token_type_ids = batch["token_type_ids"].squeeze(0)
        bbox = batch["bbox"].squeeze(0)
        x = self.model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state, maps = x.last_hidden_state, batch['maps']
        last_hidden_state = self.ln(last_hidden_state)
        reduce = self.reduce_shape(last_hidden_state, maps)
        loss_clss = self.lss 
        
        rel_s = self.rel_s(self.dropout(reduce.unsqueeze(0)))
        S = rel_s
        # S = torch.argmax(rel_s,dim =1)
        s0,s1 = S[:,:,:3,:],S[:,:,3:,:]
        # torch.argmax(s0,dim =1).numpy()
        s1 =  s1[:,:,1:-1,1:-1]#reduce
        s0 = s0[:,:,:,1:-1]# reduce



        rel_g = self.rel_g(self.dropout(reduce.unsqueeze(0)))
        # rel_s.shape
        G = rel_g
        # S = torch.argmax(rel_s,dim =1)
        g0,g1 = G[:,:,:3,:],G[:,:,3:,:]
        # torch.argmax(s0,dim =1).numpy()
        g1 =  g1[:,:,1:-1,1:-1]#reduce
        g0 = g0[:,:,:,1:-1]# reduce
        graph = torch.tensor(batch['label']).cuda()
        label = graph[0, :3, :].unsqueeze(0)
        # graph = np.array(label)

        matrix_s = graph[0, 3:, :].unsqueeze(0)
        matrix_g = graph[1, 3:, :].unsqueeze(0)
        loss_label_s = loss_clss(s0, label.long())
        loss_matrix_s = loss_clss(s1,matrix_s.long())
        loss_matrix_g = loss_clss(g1,matrix_g.long())
        loss = loss_label_s + loss_matrix_s + loss_matrix_g
        print(f'Train/loss: {loss}')
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].squeeze(0)
        attention_mask = batch["attention_mask"].squeeze(0)
        token_type_ids = batch["token_type_ids"].squeeze(0)
        bbox = batch["bbox"].squeeze(0)
        x = self.model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state, maps = x.last_hidden_state, batch['maps']
        last_hidden_state = self.ln(last_hidden_state)
        reduce = self.reduce_shape(last_hidden_state, maps)
        loss_clss = self.lss 
        
        rel_s = self.rel_s(self.dropout(reduce.unsqueeze(0)))
        S = rel_s
        s0,s1 = S[:,:,:3,:],S[:,:,3:,:]
        s1 =  s1[:,:,1:-1,1:-1]#reduce
        s0 = s0[:,:,:,1:-1]# reduce



        rel_g = self.rel_g(self.dropout(reduce.unsqueeze(0)))
        # rel_s.shape
        G = rel_g
        g0,g1 = G[:,:,:3,:],G[:,:,3:,:]
        g1 =  g1[:,:,1:-1,1:-1]#reduce
        g0 = g0[:,:,:,1:-1]# reduce
        graph = torch.tensor(batch['label']).cuda()
        # graph.copy()
        label = graph[0, :3, :].unsqueeze(0)
        # graph = np.array(label)

        
        matrix_s = graph[0, 3:, :].unsqueeze(0)
        matrix_g = graph[1, 3:, :].unsqueeze(0)
        loss_label_s = loss_clss(s0, label.long())
        pred = torch.argmax(s0,dim  =1)
        # question_heads = [i for i, ele in enumerate(pred[0]) if ele != 0]
        # answer_heads = [i for i, ele in enumerate(pred[1]) if ele != 0]
        # header_heads = [i for i, ele in enumerate(pred[2]) if ele != 0]
        
        loss_matrix_s = loss_clss(s1,matrix_s.long())
        loss_matrix_g = loss_clss(g1,matrix_g.long())
        loss = loss_label_s + loss_matrix_s + loss_matrix_g
        print(f'Val/loss: {loss}')
        return loss
