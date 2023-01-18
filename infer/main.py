from inferenceLP import LayoutParsing
import json

fi = json.load(open('cccd_raw_201.jsonl'))
PATH = '/home/grooo/Projects/eKYC/layoutlmSpade/resources/checkpoints/DP_model_finetune_15-01-2023_17-39-54.pt'
res = LayoutParsing(fi, PATH)
print(res)