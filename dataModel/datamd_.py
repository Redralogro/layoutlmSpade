import json
import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from dataModel.data_inverse import inverse
from dataModel.data_shuffle import shufflev2
trans = transforms.Compose([transforms.ToTensor()])


class DpDataSet(Dataset):
    def __init__(self, path=""):
        # self.items = []
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/layoutlm-base-uncased")
        self.items = self.get_data()

    def __len__(self):
        return len(self.items)

    def get_data(self):
        from copy import deepcopy
        root_data_ = json.load(open(self.path))
        shuffle_data = [shufflev2(item) for item in deepcopy(root_data_)]

        data1 = [{'text': item['text'], 'label':item['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord':item['coord']} for item in deepcopy(root_data_)]

        data2 = [{'text': inverse(item)['text'], 'label':inverse(item)['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord':inverse(item)['coord']} for item in deepcopy(root_data_)]

        data3 = [{'text': item['text'], 'label':item['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord': self.move_box(item)['coord']} for item in deepcopy(root_data_)]

        data4 = [{'text': inverse(item)['text'], 'label':inverse(item)['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord':self.move_box(inverse(item))['coord']} for item in deepcopy(root_data_)]

        data5 = [{'text': item['text'], 'label':item['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord':item['coord']} for item in deepcopy(shuffle_data)]

        data6 = [{'text': inverse(item)['text'], 'label':inverse(item)['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord':inverse(item)['coord']} for item in deepcopy(shuffle_data)]

        data7 = [{'text': item['text'], 'label':item['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord': self.move_box(item)['coord']} for item in deepcopy(shuffle_data)]

        data8 = [{'text': inverse(item)['text'], 'label':inverse(item)['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord':self.move_box(inverse(item))['coord']} for item in deepcopy(shuffle_data)]

        data = data1 + data2 + data3 + data4 + data5 + data6 + data7 + data8

        return data

    def __getitem__(self, idx):
        items = self.items[idx]
        input_ids, attention_mask, token_type_ids, bbox, maps = self.handle_input(
            items['text'], items['coord'], items['size'])

        return {'bbox': bbox, 'label': self.items[idx]['label'], 'maps': maps,
                'input_ids': input_ids, 'bbox': bbox, 'attention_mask': attention_mask,
                'token_type_ids': token_type_ids, 'text': self.items[idx]['text']}

    def handle_input(self, words, coord, size_):
        w, h = size_
        normalized_word_boxes = [[int(x[0][0]*1000/w), int(x[0][1]*1000/h), int(
            x[2][0]*1000/w), int(x[2][1]*1000/h)] for x in coord]
        encoding = self.tokenizer(" ".join(words), return_tensors='pt')
        token_boxes = []
        i = 0
        maps = []
        maps.extend([[i]])

        for word, box in zip(words, normalized_word_boxes):
            word_tokens = self.tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))
            i += 1
            maps.extend([[i]*len(word_tokens)])

        maps.extend([[i+1]])

        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding["token_type_ids"]
        bbox = torch.tensor([token_boxes])
        return input_ids, attention_mask, token_type_ids, bbox, maps
        """_summary_
        """

    def get_new_x_y(self, x, y, move_w, move_h, move_type):
        if move_type == 'w':
            new_x = int(x+move_w)
            new_y = int(y+0)
        if move_type == 'h':
            new_x = int(x+0)
            new_y = int(y+move_h)
        if move_type == "wh":
            new_x = int(x+move_w)
            new_y = int(y+move_h)
        return new_x, new_y

    def move_box(self, in_data, box_rate=0.1, move_rate=0.3):
        from copy import deepcopy
        data = deepcopy(in_data)
        box_id = random.choices(
            [x for x in range(len(data['coord']))], k=int(len(data['coord'])*box_rate))
        w_img = data['img_sz']['width']
        h_img = data['img_sz']['height']

        move_type = random.choice(['w', 'h', 'wh'])
        for id in box_id:

            coord = data['coord'][id]
            x, y, w, h = coord[0][0], coord[0][1], coord[2][0] - \
                coord[0][0], coord[2][1]-coord[0][1]

            move_dis_w = w*random.uniform(-move_rate, move_rate)
            move_dis_h = h*random.uniform(-move_rate, move_rate)

            new_x, new_y = self.get_new_x_y(
                x, y, move_dis_w, move_dis_h, move_type)
            x1 = max(new_x, 0)
            x2 = min(new_x+w, w_img)
            y1 = max(new_y, 0)
            y2 = min(new_y+h, h_img)

            data['coord'][id] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return data
