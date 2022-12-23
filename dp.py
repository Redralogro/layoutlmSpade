import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, BatchEncoding, BertModel,LayoutLMModel
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
config = AutoConfig.from_pretrained("microsoft/layoutlm-base-uncased")
import json
import numpy as np
from PIL import Image
from helpers import parse_input , dropout , RelationTagger

image_file = '/home/dark_hold/AI/ocr/test_images/test.jpg'
data = json.load(open('./data/dp_long.jonl.jsonl'))
result  = json.load(open('./data/raw.json'))
# np.array(data['label']).shape
print(np.array(data['label']).shape)
pa = parse_input(Image.open(image_file),
                    result['texts'],
                    result['boxes'],
                    tokenizer,
                    config,
                    data['label'],fields = data['fields'])
print(pa)
print(np.shape(pa['bbox']), np.shape(pa['actual_bbox']), np.shape(pa['labels']))


outputs = model(
    input_ids=pa['input_ids'], bbox=pa['bbox'], attention_mask=pa['attention_mask'], token_type_ids=pa['token_type_ids']

)
last_hidden_state = outputs.last_hidden_state
print(np.shape(outputs.last_hidden_state))


rel_s = RelationTagger(
            hidden_size=768,
            n_fields=3,
        )



rel_s = rel_s(dropout(last_hidden_state))

print(torch.argmax(rel_s,dim =1).shape)
print(outputs.last_hidden_state)

print(tokenizer.decode(torch.tensor([[  101,  1060]])))