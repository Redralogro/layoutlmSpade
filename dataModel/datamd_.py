import random
from functools import lru_cache
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from dataModel.helpers import handle_input, get_data, add_noise

@lru_cache
def load_tokenizer():
    return AutoTokenizer.from_pretrained(
            "microsoft/layoutlm-base-uncased")
class DpDataSet(Dataset):
    def __init__(self, path=""):
        self.path = path
        self.tokenizer = load_tokenizer()
        self.items = get_data(self.path)

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        mask_sympbol = random.choices([x for x in [" ",","]])
        items = add_noise(self.items[idx], mask_sympbol) 
        input_ids, attention_mask, token_type_ids, bbox, maps, maps_tensor = handle_input(self.tokenizer,
            items['text'], items['coord'], items['size'])

        return {'bbox': bbox, 'label': self.items[idx]['label'],
                # 'maps': maps,
                'maps_tensor': maps_tensor,
                'input_ids': input_ids, 'bbox': bbox, 'attention_mask': attention_mask,
                'token_type_ids': token_type_ids, 'text': self.items[idx]['text']}
 

