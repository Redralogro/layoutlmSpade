import torch
from helpers import infer
import json
from transformers import AutoTokenizer, AutoConfig, LayoutLMModel
from modeling.warped_model import LitLayoutParsing
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
config = AutoConfig.from_pretrained("microsoft/layoutlm-base-uncased")
PATH = './resources/checkpoints/DP_model_03-01-2023_10-47-23.pt'
modelParsing = LitLayoutParsing()
modelParsing.load_state_dict(torch.load(PATH))
modelParsing.eval()
modelParsing.cuda()


t_st = json.load(open('./data/processed/testing/20.jsonl'))


def get_bbox(jsonl_file):
    w = jsonl_file['img_sz']['width']
    h = jsonl_file['img_sz']['height']
    bboxes = [[ int(x[0][0]*1000/w),int(x[0][1]*1000/h) ,int(x[2][0]*1000/w),int(x[2][1]*1000/h)] for x in jsonl_file['coord']]
    return  torch.tensor(bboxes)

normed_bbox = get_bbox(t_st).cuda()
words = t_st['text']
with torch.no_grad():
    S,G  = modelParsing.forward((words,normed_bbox))
    s0, s1 = S[:, :, :3, :], S[:, :, 3:, :]
    g0, g1 = G[:, :, :3, :], G[:, :, 3:, :]
    text = [tokenizer.cls_token] + words + [tokenizer.sep_token]
    infer(S,G,text)


