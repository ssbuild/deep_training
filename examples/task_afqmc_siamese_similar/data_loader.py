# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 13:31
import os
import json
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper
from transformers import BertTokenizer


def pad_to_seqlength(sentence,tokenizer,max_seq_length):
    tokenizer: BertTokenizer
    o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True, )
    input_ids = np.asarray(o['input_ids'], dtype=np.int64)
    attention_mask = np.asarray(o['attention_mask'], dtype=np.int64)
    seqlen = np.asarray(len(input_ids), dtype=np.int64)
    pad_len = max_seq_length - len(input_ids)
    if pad_len > 0:
        pad_val = tokenizer.pad_token_id
        input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
        attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))

    return input_ids,attention_mask,seqlen

class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self,data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer,max_seq_length,label2id,mode = user_data
        sentence1,sentence2,label_str = data
        labels = np.asarray(1-label2id[label_str] if label_str is not None else 0, dtype=np.int64)

        input_ids, attention_mask,seqlen = pad_to_seqlength(sentence1,tokenizer,max_seq_length)
        input_ids_2, attention_mask_2, seqlen_2 = pad_to_seqlength(sentence1, tokenizer, max_seq_length)
        d = {
            'labels': labels,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'seqlen': seqlen,
            'input_ids_2': input_ids_2,
            'attention_mask_2': attention_mask_2,
            'seqlen_2': seqlen_2
        }
        return d

    #读取标签
    @staticmethod
    def read_labels_from_file(files: str):
        D = ['0','1']
        label2id = {label: i for i, label in enumerate(D)}
        id2label = {i: label for i, label in enumerate(D)}
        return label2id, id2label

    # 读取文件
    @staticmethod
    def read_data_from_file(files: typing.List,mode:str):
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    jd = json.loads(line)
                    if not jd:
                        continue
                    D.append((jd['sentence1'],jd['sentence2'], jd.get('label',None)))
        return D


    @staticmethod
    def collect_fn(batch):
        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])

        seqlen = o.pop('seqlen')
        max_len = torch.max(seqlen)

        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]

        seqlen = o.pop('seqlen_2')
        max_len = torch.max(seqlen)
        o['input_ids_2'] = o['input_ids_2'][:, :max_len]
        o['attention_mask_2'] = o['attention_mask_2'][:, :max_len]

        return o


