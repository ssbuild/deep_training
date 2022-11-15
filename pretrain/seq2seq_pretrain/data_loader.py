# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 13:31
import copy
import os
import json
import typing
import numpy as np
import torch
from asmodels.data_helper import DataHelper
from transformers import BertTokenizer

class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer,max_s_len,max_d_len,mode = user_data
        x = data
        o1 = tokenizer(x[0], max_length=max_s_len, truncation=True, add_special_tokens=True, )
        o2 = tokenizer(x[1], max_length=max_d_len, truncation=True, add_special_tokens=True, )
        input_ids = np.asarray(o1['input_ids'], dtype=np.int64)
        attention_mask = np.asarray(o1['attention_mask'], dtype=np.int64)

        slen = np.asarray(len(input_ids), dtype=np.int64)
        pad_len = max_s_len - slen
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))

        decoder_input_ids = np.asarray(o2['input_ids'], dtype=np.int64)
        labels = np.asarray(decoder_input_ids[1:],dtype=np.int64)
        decoder_input_ids = np.asarray(decoder_input_ids[:-1],dtype=np.int64)
        decoder_attention_mask = np.asarray([1] * len(decoder_input_ids),dtype=np.int64)

        dlen = np.asarray(len(decoder_input_ids), dtype=np.int64)
        pad_len = max_d_len - dlen
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            decoder_input_ids = np.pad(decoder_input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            decoder_attention_mask = np.pad(decoder_attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))

        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels':labels,
            'slen': slen,
            'dlen': dlen
        }
        return d

    # 读取文件
    @staticmethod
    def read_data_from_file(files: typing.List, mode: str):
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    jd = json.loads(line)
                    D.append((jd['content'], jd['title']))
                    if i > 1000:
                        break
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


        slen = torch.max(o.pop('slen'))
        dlen = torch.max(o.pop('dlen'))

        o['input_ids'] = o['input_ids'][:, :slen]
        o['attention_mask'] = o['attention_mask'][:, :slen]

        o['decoder_input_ids'] = o['decoder_input_ids'][:, :dlen]
        o['decoder_attention_mask'] = o['decoder_attention_mask'][:, :dlen]
        o['labels'] = o['labels'][:, :dlen]
        return o


