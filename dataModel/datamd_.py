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
        self.items = self.get_data()


    
    def __len__(self):
        return len(self.items)

    def get_data(self):
        return [{'text':item['text'], 'label':item['label'],
        'size':[int(item['img_sz']['width']),
                int(item['img_sz']['height'])],
        'coord':item['coord'] } for item in json.load(open(self.path))]

    def __getitem__(self,idx):
        bbox,normalized_word_boxes = self.handle_input(
            self.items[idx]['text'], self.items[idx]['coord'], self.items[idx]['size'])
        return {'bbox':bbox ,'label':self.items[idx]['label'],
                'text': self.items[idx]['text'], 'normalized_word_boxes': normalized_word_boxes}

    
    def handle_input(self,words, coord, size_):
        w,h = size_
        normalized_word_boxes = [[ int(x[0][0]*1000/w),int(x[0][1]*1000/h) ,int(x[2][0]*1000/w),int(x[2][1]*1000/h)] for x in coord]
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

        bbox = torch.tensor([token_boxes])
        return bbox, torch.tensor(normalized_word_boxes)
