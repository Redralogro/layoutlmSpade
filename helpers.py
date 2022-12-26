from graph_stuff import *  
from traceback import print_exc
import numpy as np
import torch.nn as nn
from typing import Dict
import json
import yaml

dropout = nn.Dropout(0.1)
class RelationTagger(nn.Module):
    def __init__(self, n_fields, hidden_size, head_p_dropout=0.1):
        super().__init__()
        self.head = nn.Linear(hidden_size, hidden_size)
        self.tail = nn.Linear(hidden_size, hidden_size)
        self.field_embeddings = nn.Parameter(
            torch.rand(1, n_fields, hidden_size))
        self.W_label_0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_label_1 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, enc):

        enc_head = self.head(enc)
        enc_tail = self.tail(enc)

        batch_size = enc_tail.size(0)
        field_embeddings = self.field_embeddings.expand(batch_size, -1, -1)
        enc_head = torch.cat([field_embeddings, enc_head], dim=1)

        score_0 = torch.matmul(
            enc_head, self.W_label_0(enc_tail).transpose(1, 2))
        score_1 = torch.matmul(
            enc_head, self.W_label_1(enc_tail).transpose(1, 2))

        score = torch.cat(
            [
                score_0.unsqueeze(1),
                score_1.unsqueeze(1),
            ],
            dim=1,
        )
        return score



def tensorize(x):
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x)
    else:
        try:
            return torch.as_tensor(np.array(x))
        except Exception:
            print_exc()
            return torch.tensor(x)

def normalize_box(box, width, height):
    # x1, y1, x2, y2
    normed = [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]
    return np.clip(normed, 0, 1000)


def load_config(path: str) -> Dict:
    if path.endswith("yml"):
        with open(path,"r") as f:
            config = yaml.load(stream=f, Loader=yaml.FullLoader)
    elif path.endswith("json"):
        with open(path,"r") as f:
            config = json.load(stream=f)
    elif path.endswith("yaml"):
        with open(path,"r") as f:
            config = yaml.load(stream=f, Loader=yaml.FullLoader)
    else:
        raise "Invalid file format"
    return config

def parse_input(
    image,
    words,
    actual_boxes,
    tokenizer,
    config,
    label,
    fields,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
    pad_token_box=[0, 0, 0, 0],
):
    width, height = image.size
    boxes = [normalize_box(b, width, height) for b in actual_boxes]
    if label is not None:
        label = tensorize(label)
        token_map = map_token(tokenizer, words, offset=len(fields))
        rel_s = tensorize(label[0])
        rel_g = tensorize(label[1])
        token_rel_s = expand(rel_s, token_map, lh2ft=True,
                               in_tail=True, in_head=True)
        token_rel_g = expand(rel_g, token_map, fh2ft=True)
        label = torch.cat(
            [token_rel_s.unsqueeze(0), token_rel_g.unsqueeze(0)],
            dim=0,
        )
        print ("-----------label-------")
        print (label)

    tokens = []
    token_boxes = []
    actual_bboxes = []  # we use an extra b because actual_boxes is already used
    token_actual_boxes = []
    are_box_first_tokens = []
    maps = []
    i = 0
    for word, box, actual_bbox in zip(words, boxes, actual_boxes):
        word_tokens = tokenizer.tokenize(word)
        # print(word_tokens)
        tokens.extend(word_tokens)
        token_boxes.extend([box] * len(word_tokens))
        maps.extend([i]*len(word_tokens))
        i += 1
        actual_bboxes.extend([actual_bbox] * len(word_tokens))
        token_actual_boxes.extend([actual_bbox] * len(word_tokens))
        are_box_first_tokens.extend([1] + [0] * (len(word_tokens) - 1))
    print(np.array(maps))
    # Truncation: account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    true_length = config.max_position_embeddings - special_tokens_count
    if len(tokens) > true_length:
        tokens = tokens[:true_length]
        token_boxes = token_boxes[:true_length]
        token_actual_boxes = token_actual_boxes[:true_length]
        actual_bboxes = actual_bboxes[:true_length]
        are_box_first_tokens = are_box_first_tokens[:true_length]
        if label is not None:
            label = label[:, : (len(fields) + true_length), :true_length]

    # add [SEP] token, with corresponding token boxes and actual boxes
    tokens += [tokenizer.sep_token]
    token_boxes += [sep_token_box]
    actual_bboxes += [[0, 0, width, height]]
    token_actual_boxes += [[0, 0, width, height]]
    are_box_first_tokens += [1]
    # use labels for auxilary result
    if label is not None:
        n, i, j = label.shape
        labels = torch.zeros((n, i + 1, j + 1), dtype=label.dtype)
        labels[:, :i, :j] = label
        label = labels

    segment_ids = [0] * len(tokens)

    # print("----")
    # edge_0 = []
    # for (i, j) in zip(*torch.where(token_rel_s)):
    #     l = [(fields + tokens)[i], tokens[j]]
    #     print(l)
    #     edge_0.append(" -> ".join(l))

    # next: [CLS] token
    tokens = [tokenizer.cls_token] + tokens
    token_boxes = [cls_token_box] + token_boxes
    actual_bboxes = [[0, 0, width, height]] + actual_bboxes
    token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
    segment_ids = [1] + segment_ids
    are_box_first_tokens = [2] + are_box_first_tokens
    # This is tricky because cls need to be inserted
    # after the labels
    if label is not None:
        nfields = len(fields)
        top_half = label[:, :nfields, :]
        bottom_half = label[:, nfields:, :]
        n, i, j = label.shape
        new_label = torch.zeros(n, i + 1, j + 1, dtype=label.dtype)
        new_label[:, :nfields, 1:] = top_half
        new_label[:, (nfields + 1):, 1:] = bottom_half
        label = new_label

    #     print("----")
    #     print("AFter CLS")
    #     edge_1 = []
    #     for (i, j) in zip(*torch.where(label[0])):
    #         l = [(fields + tokens)[i], tokens[j]]
    #         print(l)
    #         edge_1.append(" -> ".join(l))
    #     print("----")

    #     for (i, j) in zip(edge_0, edge_1):
    #         print(i, j, i == j)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = config.max_position_embeddings - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    input_mask += [0] * padding_length
    segment_ids += [tokenizer.pad_token_id] * padding_length
    token_boxes += [pad_token_box] * padding_length
    token_actual_boxes += [pad_token_box] * padding_length
    are_box_first_tokens += [3] * padding_length

    assert len(input_ids) == config.max_position_embeddings
    assert len(input_mask) == config.max_position_embeddings
    assert len(segment_ids) == config.max_position_embeddings
    assert len(token_boxes) == config.max_position_embeddings
    assert len(token_actual_boxes) == config.max_position_embeddings
    assert len(are_box_first_tokens) == config.max_position_embeddings
    
    # Label parsing
    if label is not None:
        n, i, j = label.shape
        labels = torch.zeros(
            n,
            config.max_position_embeddings + len(fields),
            config.max_position_embeddings,
            dtype=label.dtype,
        )
        labels[:, :i, :j] = label
        label = labels

    # assert itc_labels.shape[0] == config.max_position_embeddings
    # assert stc_labels.shape[1] == config.max_position_embeddings
    # assert stc_labels.shape[2] == config.max_position_embeddings

    # labels = torch.cat(
    #     [torch.zeros(labels.shape[0], 1, config.max_position_embeddings), labels],
    #     dim=1,
    # )

    # The unsqueezed dim is the batch dim for each type
    if label is not None:
        return {
            "text_tokens": tokens,
            "input_ids": tensorize(input_ids).unsqueeze(0),
            "attention_mask": tensorize(input_mask).unsqueeze(0),
            "token_type_ids": tensorize(segment_ids).unsqueeze(0),
            "bbox": tensorize(token_boxes).unsqueeze(0),
            "actual_bbox": tensorize(token_actual_boxes).unsqueeze(0),
            # "itc_labels": itc_labels.unsqueeze(0),
            # "stc_labels": stc_labels.unsqueeze(0),
            "labels": tensorize(labels).unsqueeze(0),
            "are_box_first_tokens": tensorize(are_box_first_tokens).unsqueeze(0),
        }
    else:
        return {
            "text_tokens": tokens,
            "input_ids": tensorize(input_ids).unsqueeze(0),
            "attention_mask": tensorize(input_mask).unsqueeze(0),
            "token_type_ids": tensorize(segment_ids).unsqueeze(0),
            "bbox": tensorize(token_boxes).unsqueeze(0),
            "actual_bbox": tensorize(token_actual_boxes).unsqueeze(0),
            "are_box_first_tokens": tensorize(are_box_first_tokens).unsqueeze(0),
        }
