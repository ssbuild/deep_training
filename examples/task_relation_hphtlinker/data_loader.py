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
    def on_data_process(self, data: typing.Any, user_data: tuple):

        tokenizer: BertTokenizer
        tokenizer,max_seq_length,label2id,mode = user_data
        sentence,spo_list = data

        sub_text = []
        for char in sentence:
            sub_text.append(char)

        tok_to_orig_start_index = []
        tok_to_orig_end_index = []
        orig_to_tok_index = []
        tokens = []
        text_tmp = ''

        for (i, token) in enumerate(sub_text):
            orig_to_tok_index.append(len(tokens))
            sub_tokens = tokenizer._tokenize(token)
            text_tmp += token

            if len(sub_tokens) != 1:
                print('!!! bad token')
                sub_tokens = [tokenizer.unk_token]
            flag = False
            for sub_token in sub_tokens:
                tok_to_orig_start_index.append(len(text_tmp) - len(token))
                tok_to_orig_end_index.append(len(text_tmp) - 1)
                tokens.append(sub_token)
                if len(tokens) >= max_seq_length - 2:
                    flag = True
                    break
            if flag:
                break

        seq_len = len(tokens)

        spoes = {}
        for s, p, o in d['spo_list']:
            s = tokenizer.encode(s)[0][1:-1]
            p = predicate2id[p]
            o = tokenizer.encode(o)[0][1:-1]
            s_idx = search(s, token_ids)
            o_idx = search(o, token_ids)
            if s_idx != -1 and o_idx != -1:
                s = (s_idx, s_idx + len(s) - 1)
                o = (o_idx, o_idx + len(o) - 1, p)
                if s not in spoes:
                    spoes[s] = []
                spoes[s].append(o)


        # add [CLS] and [SEP] token, they are tagged into "O" for outside
        if seq_len > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            tok_to_orig_start_index = tok_to_orig_start_index[0:(max_seq_length - 2)]
            tok_to_orig_end_index = tok_to_orig_end_index[0:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_mask = [1] * len(tokens)
        # "O" tag for [PAD], [CLS], [SEP] token
        outside_label = [[1] + [0] * (num_labels - 1)]

        labels = outside_label + labels + outside_label
        tok_to_orig_start_index = [-1] + tok_to_orig_start_index + [-1]
        tok_to_orig_end_index = [-1] + tok_to_orig_end_index + [-1]
        if seq_len < max_seq_length:
            tokens = tokens + ["[PAD]"] * (max_seq_length - seq_len - 2)
            input_mask = input_mask + [0] * (max_seq_length - seq_len - 2)
            labels = labels + outside_label * (max_seq_length - len(labels))
            tok_to_orig_start_index = tok_to_orig_start_index + [-1] * (max_seq_length - len(tok_to_orig_start_index))
            tok_to_orig_end_index = tok_to_orig_end_index + [-1] * (max_seq_length - len(tok_to_orig_end_index))

        token_ids = np.array(tokenizer.convert_tokens_to_ids(tokens))

        mask = np.logical_and(token_ids != tokenizer.pad_token_id,
                              np.logical_and(token_ids != tokenizer.cls_token_id, token_ids != tokenizer.sep_token_id))

        d = {
            "input_ids": np.array(token_ids),
            "attention_mask": np.asarray(input_mask),
            "mask": np.asarray(mask),
            "labels": np.array(labels),
            "seq_len": np.array(seq_len),
            # "tok_to_orig_start_index": np.array(tok_to_orig_start_index),
            # "tok_to_orig_end_index": np.array(tok_to_orig_end_index),

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

        o['labels'] = o['labels'][:,:, :max_len,:max_len]
        return o


