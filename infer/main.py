from inferenceLP import LayoutParsing
import json

fi = json.load(open('/home/grooo/Projects/eKYC/layoutlmSpade/data/cccd_raw/cccd_raw_201.jsonl'))
PATH = '/home/grooo/Projects/eKYC/layoutlmSpade/resources/checkpoints/DP_model_finetune_20-01-2023_03-32-22.pt'
res = LayoutParsing(fi, PATH)
print(res)

