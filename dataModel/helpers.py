import torch
import random
import json
import numpy as np
from typing import List
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def add_noise(data,mask_sympbol):
        text_id = random.choices([x for x in range(len(data['text']))], k=int(len(data['text'])*0.1))
        for idx in text_id:
            t = data['text'][idx] + ' '
            data['text'][idx] = t 
            # print(t)
        return data


def handle_input(tokenizer, words, coord, size_):
    w, h = size_
    normalized_word_boxes = [[int(x[0][0]*1000/w), int(x[0][1]*1000/h), int(
        x[2][0]*1000/w), int(x[2][1]*1000/h)] for x in coord]
    encoding = tokenizer(" ".join(words), return_tensors='pt')
    token_boxes = []
    i = 0
    maps_tensor = []
    maps = []
    maps.extend([[i]])
    maps_tensor.extend([i])

    for word, box in zip(words, normalized_word_boxes):
        word_tokens = tokenizer.tokenize(word)
        token_boxes.extend([box] * len(word_tokens))
        i += 1
        if len(word_tokens) > 0:
            maps.extend([[i]*len(word_tokens)])
        maps_tensor.extend([i]*len(word_tokens))

    maps.extend([[i+1]])
    maps_tensor.extend([i+1])

    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]
    bbox = torch.tensor([token_boxes])
    maps_tensor = torch.tensor(maps_tensor).unsqueeze(0)
    return input_ids, attention_mask, token_type_ids, bbox, maps, maps_tensor
    """_summary_
    """

def handle_input_v2(tokenizer, words, coord, size_):
    w, h = size_
    normalized_word_boxes = [[int(x[0][0]*1000/w), int(x[0][1]*1000/h), int(
        x[2][0]*1000/w), int(x[2][1]*1000/h)] for x in coord]
    encoding = tokenizer(" ".join(words), return_tensors='pt')
    token_boxes = []
    i = 0
    
    maps = []
    maps.extend([i])

    for word, box in zip(words, normalized_word_boxes):
        word_tokens = tokenizer.tokenize(word)
        token_boxes.extend([box] * len(word_tokens))
        i += 1
        maps.extend([i]*len(word_tokens))

    maps.extend([i+1])

    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
    #layoutlm input
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]
    bbox = torch.tensor([token_boxes])
    #mapping bboxes
    maps = torch.tensor(maps).unsqueeze(0)
    
    return input_ids, attention_mask, token_type_ids, bbox, maps
    """_summary_
    """

def get_bbox(jsonl_file):
    bboxes = [[ int(x[0][0]),int(x[0][1]) ,int(x[2][0]),int(x[2][1])] for x in jsonl_file['coord']]
    return  bboxes

def get_new_x_y( x, y, move_w, move_h, move_type):
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


def move_box( in_data, box_rate=0.1, move_rate=0.3):
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

        new_x, new_y = get_new_x_y(
            x, y, move_dis_w, move_dis_h, move_type)
        x1 = max(new_x, 0)
        x2 = min(new_x+w, w_img)
        y1 = max(new_y, 0)
        y2 = min(new_y+h, h_img)

        data['coord'][id] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return data

#====================================================================================

def rand_perm(n: int):
    perm = list(range(n))
    random.shuffle(perm)
    return perm
    
def sort_perm(arr: List, perm: List[int]):
    return [arr[i] for i in perm]


def sort_perm2d(label: np.ndarray, perm):
    m, n = label.shape
    n_fields = m - n
    # permutation offset
    row_perm = list(range(n_fields)) + [p + n_fields for p in perm]
    col_perm = perm
    idx = np.ix_(row_perm, col_perm)
    return label[idx]

def shufflev2(f):
    n= len(f['text'])
    idx_list = list(range(n))
    idx_list = rand_perm(n)
    text = sort_perm(f['text'], idx_list)
    coord = sort_perm(f['coord'], idx_list)
    f['text'] = text
    f['coord'] = coord
    graph = np.array(f['label'])
    s_matrix = graph[0]
    g_matrix = graph[1]
    #matrix
    tmp = sort_perm2d(s_matrix, idx_list).tolist()
    tmpG = sort_perm2d(g_matrix, idx_list).tolist()

    for i in range(len(f['label'][0])):
            # print(i)
            f['label'][0][i] = tmp[i]
            f['label'][1][i] = tmpG[i]

    return f



def inverse_boxes(boxes):
    bboxes = []
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        bboxes.insert(0, [x1, y1, x2, y2])

    return bboxes

def inverse(f):
    f['label'][0][0] = f['label'][0][0][::-1]
    f['label'][0][1] = f['label'][0][1][::-1]
    f['label'][0][2] = f['label'][0][2][::-1]
    f['label'][1][0] = f['label'][1][0][::-1]
    f['label'][1][1] = f['label'][1][1][::-1]
    f['label'][1][2] = f['label'][1][2][::-1]

    tmp =[]
    tmpG =[]
    for i in range(len(f['label'][0])-1,0,-1):
        if i > 2: 
            tmp.append(f['label'][0][i][::-1])
            tmpG.append(f['label'][1][i][::-1])

    for i in range(3,len(f['label'][0]),1):
        # print(i)
        f['label'][0][i] = tmp[i-3]
        f['label'][1][i] = tmpG[i-3]
    

    f['text'] = f['text'][::-1] 
    f['coord'] = [[[i[0],i[1]], [i[2],i[1]], [i[2], i[3]], [i[0], i[3]]]  for i in inverse_boxes(get_bbox(f))]

    return f


def add_noise(data,mask_sympbol):
        text_id = random.choices([x for x in range(len(data['text']))], k=int(len(data['text'])*0.1))
        for idx in text_id:
            t = data['text'][idx] + ' '
            data['text'][idx] = t 
            # print(t)
        return data

#####################################################


def get_data(PATH):
        from copy import deepcopy
        # with open(self.path,'r', encoding ='utf8') as f:
        f = open(PATH, 'r', encoding='utf8')
        root_data_ = [json.loads(s) for s in f.readlines()]
        f.close()
        # with open(PATH, 'r', encoding='utf8') as f:
        #     root_data_ = [json.loads(s) for s in f.readlines()]
        # root_data_ = json.load(open(self.path))
        shuffle_data = [shufflev2(item) for item in deepcopy(root_data_)]

        data1 = [{'text': item['text'], 'label':item['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord':item['coord']} for item in deepcopy(root_data_)]

        data2 = [{'text': inverse(item)['text'], 'label':inverse(item)['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord':inverse(item)['coord']} for item in deepcopy(root_data_)]

        data3 = [{'text': item['text'], 'label':item['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord': move_box(item)['coord']} for item in deepcopy(root_data_)]

        data4 = [{'text': inverse(item)['text'], 'label':inverse(item)['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord':move_box(inverse(item))['coord']} for item in deepcopy(root_data_)]

        data5 = [{'text': item['text'], 'label':item['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord':item['coord']} for item in deepcopy(shuffle_data)]

        data6 = [{'text': inverse(item)['text'], 'label':inverse(item)['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord':inverse(item)['coord']} for item in deepcopy(shuffle_data)]

        data7 = [{'text': item['text'], 'label':item['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord': move_box(item)['coord']} for item in deepcopy(shuffle_data)]

        data8 = [{'text': inverse(item)['text'], 'label':inverse(item)['label'],
                 'size':[int(item['img_sz']['width']), int(item['img_sz']['height'])],
                  'coord':move_box(inverse(item))['coord']} for item in deepcopy(shuffle_data)]

        data = data1  + data2 + data3 + data4 + data5 + data6 + data7 + data8

        return data
