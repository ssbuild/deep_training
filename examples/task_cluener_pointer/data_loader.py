# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 13:31
import os
import json
import typing
import numpy as np
import torch
from asmodels.data_helper import DataHelper
from transformers import BertTokenizer

class NER_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data_index: int, data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer,max_seq_length,label2id = user_data
        sentence,label_dict = data

        o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True, )
        input_ids = np.asarray(o['input_ids'], dtype=np.int64)
        attention_mask = np.asarray(o['attention_mask'], dtype=np.int64)
        seqlen = np.asarray(len(input_ids), dtype=np.int64)

        labels = np.zeros(shape=(len(label2id),max_seq_length,max_seq_length),dtype=np.int64)

        real_label = []
        if label_dict is not None:
            for label_str, o in label_dict.items():
                pts = list(o.values())[0]
                labelid = label2id[label_str]
                for pt in pts:
                    if pt[1] < max_seq_length:
                        labels[labelid,pt[0],pt[1]] = 1
                    real_label.append((labelid,pt[0],pt[1]))

        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'seqlen': seqlen,
            'real_label': np.asarray(bytes(json.dumps(real_label,ensure_ascii=False),encoding='utf-8'))
        }
        return d

    #读取标签
    @staticmethod
    def read_labels_from_file(label_fname: str):
        labels = [
            'address','book','company','game','government','movie','name','organization','position','scene'
        ]
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
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
                    D.append((jd['text'], jd.get('label',None)))
        return D[:1000]


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
        # max_len = torch.max(seqlen)
        #
        # o['input_ids'] = o['input_ids'][:, :max_len]
        # o['attention_mask'] = o['attention_mask'][:, :max_len]
        # if 'token_type_ids' in o:
        #     o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        return o


