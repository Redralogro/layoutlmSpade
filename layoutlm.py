import numpy as np
import torch
from torch import nn
from transformers import (AutoConfig, AutoModel, AutoTokenizer, BatchEncoding,
                          BertModel, LayoutLMModel)


class layoutlmBase(nn.Module):
    def __init__(self):
        super(layoutlmBase, self).__init__()
        self.model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
        self.tokenizer  = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        self.__freeze_bert_model__()

    def __freeze_bert_model__(self):
        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, data ):
        # words, bboxes  =  data 
        # # print(words,bboxes)
        # data = self.handle_input(words, bboxes)
        print(data)
        x = self.model(
            input_ids=data['input_ids'], bbox=data['bbox'], attention_mask=data['attention_mask'], token_type_ids=data['token_type_ids']
        )
        return x.last_hidden_state, data['maps']