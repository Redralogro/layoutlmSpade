import json
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

trans = transforms.Compose([transforms.ToTensor()])

class DpDataSet(Dataset):
    def __init__(self,path = "" ):
        # self.items = []
        self.path = path
        self.tokenizer  = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        self.jsons = self.get_jsonl()
        self.items = self.get_data()

        
    def get_jsonl(self):
        # dir_list = os.listdir(self.path)
        # dir_id_list =  [self.path+'/'+ x for x in dir_list]
        # id_list = [ json.load(open(f))  for f in dir_id_list]
        id_list = json.load(open(self.path))

        return id_list
    
    def __len__(self):
        return len(self.items)

    def get_data(self):
        return [{'text':item['text'], 'label':item['label'],
        'size':[int(item['img_sz']['width']),
                int(item['img_sz']['height'])],
        'coord':item['coord'] } for item in self.jsons]

    def __getitem__(self,idx):
        input_ids,attention_mask,token_type_ids,bbox,maps = self.handle_input(
            self.items[idx]['text'], self.items[idx]['coord'], self.items[idx]['size'])
        # sample = {'text':text, 'label':label, 'size': size_img, 'coord':coord }
        return {'input_ids': input_ids, 'attention_mask': attention_mask,'token_type_ids':token_type_ids,'bbox':bbox , 'maps': maps ,'label':self.items[idx]['label']}

    
    def handle_input(self,words, coord, size_):
        w,h = size_
        normalized_word_boxes = tuple([[ int(x[0][0]*1000/w),int(x[0][1]*1000/h) ,int(x[2][0]*1000/w),int(x[2][1]*1000/h)] for x in coord])
        encoding = self.tokenizer(" ".join(words), return_tensors='pt')
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

        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding["token_type_ids"]
        bbox = torch.tensor([token_boxes])
        return input_ids,  attention_mask,token_type_ids,bbox , maps
