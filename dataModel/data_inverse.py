import sys
sys.path.append('..')

import json
import numpy as np


def get_bbox(jsonl_file):
    bboxes = [[ int(x[0][0]),int(x[0][1]) ,int(x[2][0]),int(x[2][1])] for x in jsonl_file['coord']]
    return  bboxes

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