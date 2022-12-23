import torch
import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, BatchEncoding, BertModel,LayoutLMModel

class layoutlmBase(nn.Module):
    def __init__(self):
        super(layoutlmBase, self).__init__()
        self.model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
        self.tokenizer  = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        self.__freeze_bert_model__()

    def __freeze_bert_model__(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def handle_input(self,words, bboxes):
        normalized_word_boxes = bboxes
        encoding = self.tokenizer(" ".join(words), return_tensors="pt")
        token_boxes = []
        tokens = []
        tokens = [self.tokenizer.cls_token] + tokens
        i =0
        maps = []
        maps.extend([[i]])
        for word, box in zip(words, normalized_word_boxes):
            word_tokens = self.tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))
            tokens.extend(word_tokens)
            i += 1
            maps.extend([[i]*len(word_tokens)])

        tokens += [self.tokenizer.sep_token]
        maps.extend([[i+1]])

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding["token_type_ids"]
        bbox = torch.tensor([token_boxes])
        return {'input_ids': input_ids, 'attention_mask': attention_mask,'token_type_ids':token_type_ids,'bbox':bbox , 'maps': maps}

    def forward(self,words, bboxes ):
        inputs = self.handle_input(words, bboxes)
        x = self.model(
            input_ids=inputs['input_ids'], bbox=inputs['bbox'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids']
        )
        return x.last_hidden_state, inputs['maps']