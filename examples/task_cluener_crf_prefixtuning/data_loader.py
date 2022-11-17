# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 13:31
import os
import json
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper
from transformers import BertTokenizer #仅编写提示使用，实际不一定使用

class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):

        tokenizer: BertTokenizer
        tokenizer,max_seq_length,label2id,mode = user_data
        sentence,label_dict = data

        input_ids = tokenizer.convert_tokens_to_ids(list(sentence))
        if len(input_ids) > max_seq_length - 2:
            input_ids = input_ids[:max_seq_length - 2]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        input_ids = np.asarray(input_ids, dtype=np.int64)
        attention_mask = np.asarray(attention_mask, dtype=np.int64)
        seqlen = np.asarray(len(input_ids), dtype=np.int64)

        labels = np.zeros(shape=(seqlen,),dtype=np.int64)

        for label_str, o in label_dict.items():
            pts = list(o.values())[0]
            for pt in pts:
                if pt[1] > seqlen - 2:
                    continue
                pt[0] += 1
                pt[1] += 1

                span_len = pt[1] - pt[0] + 1
                if span_len == 1:
                    labels[pt[0]] = label2id['S_' + label_str]
                elif span_len == 2:
                    labels[pt[0]] = label2id['B_' + label_str]
                    labels[pt[1]] = label2id['E_' + label_str]
                    for i in range(span_len - 2):
                        labels[pt[0] + 1 + i] = label2id['I_' + label_str]


        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'seqlen': seqlen,
        }
        return d

    #读取标签
    @staticmethod
    def read_labels_from_file(label_fname: str):
        labels = [
            'address','book','company','game','government','movie','name','organization','position','scene'
        ]
        labels = ['O'] + [t + '_' + l  for t in ['B','I','E','S'] for l in labels]
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
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]

        o['labels'] = o['labels'][:,:max_len]
        return o


