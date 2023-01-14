import json
import numpy as np
import random
from typing import List


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


def get_bbox(jsonl_file):
    bboxes = [[ int(x[0][0]),int(x[0][1]) ,int(x[2][0]),int(x[2][1])] for x in jsonl_file['coord']]
    return  bboxes


def shuffe(l:list):
    random.seed(26)
    random.shuffle(l)
    return l

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

def shuffle(f):
    data_t = list(zip(f['text'], f['coord']))

    data_t = shuffe(data_t)
    text, coord = zip(*data_t)

    f['text'] = text
    f['coord'] = coord

    f['label'][0][0] = shuffe(f['label'][0][0])
    f['label'][0][1] = shuffe(f['label'][0][1])
    f['label'][0][2] = shuffe(f['label'][0][2])
    f['label'][1][0] = shuffe(f['label'][1][0])
    f['label'][1][1] = shuffe(f['label'][1][1])
    f['label'][1][2] = shuffe(f['label'][1][2])
    tmp =[]
    tmpG =[]
    for i in range(len(f['label'][0])):
        if i > 2: 
            tmp.append(shuffe(f['label'][0][i]))
            tmpG.append(shuffe(f['label'][1][i]))
            # tmpG.append(f['label'][1][i][::-1])
    tmp = shuffe(tmp)
    tmpG = shuffe(tmpG)
    for i in range(3,len(f['label'][0]),1):
        # print(i)
        f['label'][0][i] = tmp[i-3]
        f['label'][1][i] = tmpG[i-3]
    
    return f