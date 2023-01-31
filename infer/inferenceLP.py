import json
import networkx as nx
import torch
from infer.model.base_model import LitBaseParsing
from transformers import AutoTokenizer
import numpy as np
from functools import lru_cache
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

def get_strings(heads, data: list, graph): 
    temp = []
    # G = nx.Graph(graph_s[0,3:,:]) # s
    G = nx.Graph(graph) # s
    try:
        for index in heads:
            dfs = list(nx.dfs_edges(G, source=int(index)))
            dfs
            if  dfs == []:
                header = [int(index)]
            else: header =  [dfs[0][0]] + [x[1]  for i,x in enumerate (dfs)]
            str_ = ''
            for i in header:
                str_ += ' ' + data[int(i)] 
                assert i <= len(data)
            temp.append([index, str_[1:]])
    except Exception:
        pass
    return temp

def get_bbox(jsonl_file):
    w = jsonl_file['img_sz']['width']
    h = jsonl_file['img_sz']['height']
    bboxes = [[ int(x[0][0]*1000/w),int(x[0][1]*1000/h) ,int(x[2][0]*1000/w),int(x[2][1]*1000/h)] for x in jsonl_file['coord']]
    return  torch.tensor(bboxes)

@lru_cache
def LoadModel(path):
    PATH = path
    modelParsing = LitBaseParsing()
    modelParsing.load_state_dict(torch.load(PATH))
    modelParsing.eval()
    modelParsing.cuda()
    
    return modelParsing

def LayoutParsing(jsonfile, path):
    
    modelParsing = LoadModel(path)
    
    words = jsonfile['text']
    normalized_word_boxes = get_bbox(jsonfile)
    encoding = tokenizer(" ".join(words), return_tensors='pt')
    token_boxes = []
    i = 0
    maps = []
    maps.extend([[i]])
    result = {}
    for word, box in zip(words, normalized_word_boxes):
        word_tokens = tokenizer.tokenize(word)
        token_boxes.extend([box] * len(word_tokens))
        i += 1
        maps.extend([[i]*len(word_tokens)])

    maps.extend([[i+1]])

    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
    input_ids = encoding["input_ids"].cuda()
    attention_mask = encoding["attention_mask"].cuda()
    token_type_ids = encoding["token_type_ids"].cuda()
    bbox = torch.tensor([token_boxes]).cuda()
    
    with torch.no_grad():
        
        # print(input_ids.shape, attention_mask.shape, token_type_ids.shape, bbox.shape,np.shape(maps))
        S,G  = modelParsing((input_ids, attention_mask, token_type_ids, bbox, maps))
        print(np.shape(S), np.shape(G))
        s0, s1 = S[:, :, :3, :], S[:, :, 3:, :]
        g0, g1 = G[:, :, :3, :], G[:, :, 3:, :]
        pred_matrix_s = torch.softmax(s1, dim=1)
        pred_matrix_s = torch.argmax(pred_matrix_s, dim=1).squeeze(0)
        pred_matrix_g = torch.softmax(g1, dim=1)
        pred_matrix_g = torch.argmax(pred_matrix_g, dim=1).squeeze(0)
        pred_label = torch.argmax(s0, dim=1).squeeze(0)
        pred_group = torch.argmax(g0, dim=1).squeeze(0)
        s_map,g_map,s_head,g_head = pred_matrix_s[1:-1,1:-1] ,pred_matrix_g[1:-1,1:-1],pred_label[:,1:-1],pred_group[:,1:-1]

        t_s = torch.cat((s_head,s_map),0).unsqueeze(0)
        t_g = torch.cat((g_head,g_map),0).unsqueeze(0)
        label_predict = torch.cat((t_s,t_g),0).cpu()
        label_predict = label_predict.tolist()
        # text = [tokenizer.cls_token] + words + [tokenizer.sep_token]
        # infer(S, G, text)
        
        
        graph = np.array(label_predict)
        S_ = graph[0, 3:, :]
        G_ = graph[1, 3:, :]
        s0_ac, s1_ac =  graph[0, :3, :], graph[0, 3:, :]
        g0_ac, g1_ac =  graph[1, :3, :], graph[1, 3:, :]
        # graph.shape
        question_heads = [i for i, ele in enumerate(s0_ac[0]) if ele != 0]
        answer_heads = [i for i, ele in enumerate(s0_ac[1]) if ele != 0]
        ques = get_strings(question_heads, words, S_)
        ans = get_strings(answer_heads, words, S_)
        # print(f'[GROUND TRUTH]: Ques:{ques} \n Ans: {ans}')
        # # print(f'[GROUND TRUTH]')
        for ans_dix in answer_heads:
        # for ans_dix in question_heads:

            G_pred = nx.Graph(g1_ac)  # group
            dfs = list(nx.dfs_edges(G_pred, source=int(ans_dix)))
            # print(dfs)

            if len(dfs) ==0:
                result[str(ans_dix)] = get_strings([ans_dix ], words, S_)[0][1]
                # print(get_strings([ans_dix ], words, S_)[0][1])
            if len(dfs) != 0:
                a, q = dfs[0]
                qu_s = [qs[1] for qs in ques if q in qs]
                an_s = [as_[1] for as_ in ans if a in as_]
                # if len(qu_s)== len(an_s):
                #     print(qu_s[0], an_s[0])
                
                try:
                    # print(f'{qu_s[0]}|{an_s[0]}')
                    result [str(qu_s[0])] = an_s[0]
                except Exception:
                    if len(qu_s) ==0 and len(an_s) !=0:
                        # print(f'-|{an_s[0]}') 
                        result [str(ans_dix)] = an_s[0]
                    if len(qu_s) !=0 and len(an_s) ==0:
                        # print(f'{qu_s[0]}|-')
                        continue
    
    return result




