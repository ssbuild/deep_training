# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 13:31
import os
import json
import typing
import numpy as np
import torch
from asmodels.data_helper import DataHelper
from transformers import BertTokenizer


def parse_label(spo_list, label2id, tokens, max_length):
    # 2 tags for each predicate + I tag + O tag
    num_labels = 2 * (len(label2id.keys()) - 2) + 2
    seq_len = len(tokens)
    # initialize tag
    labels = [[0] * num_labels for i in range(seq_len)]

    for spo in spo_list:
        for relation_name in spo:
            relation = spo[relation_name]
            if len(relation) != 2:
                raise Exception('len(relation) != 2')
            subject = relation[0]
            object = relation[1]

            if relation_name not in label2id.keys():
                relation_name = relation_name + '_' + object['label']
                if relation_name not in label2id.keys():
                    raise relation_name + ' not in ' + ''.join(list(label2id.keys()))

            subject_ids = [subject['pos'][0],subject['pos'][1] - 1]
            object_ids= [object['pos'][0],object['pos'][1] - 1]

            if subject_ids[1] >= max_length -2 or object_ids[1] >= max_length -2:
                continue

            label_subject = label2id[relation_name]
            label_object = label_subject + len(label2id.keys()) - 2


            index = subject_ids[0]
            subject_tokens_len = subject_ids[1] - subject_ids[0] + 1
            labels[index][label_subject] = 1
            for i in range(subject_tokens_len - 1):
                labels[index + i + 1][1] = 1

            index = object_ids[0]
            labels[index][label_object] = 1
            object_tokens_len = object_ids[1] - object_ids[0] + 1

            if index + object_tokens_len - 1 >= seq_len:
                print(spo_list)
                print(seq_len,object_ids,object_tokens_len)
            for i in range(object_tokens_len - 1):
                labels[index + i + 1][1] = 1

    # if token wasn't assigned as any "B"/"I" tag, give it an "O" tag for outside
    for i in range(seq_len):
        if labels[i] == [0] * num_labels:
            labels[i][0] = 1

    return labels



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
        # 2 tags for each predicate + I tag + O tag
        num_labels = 2 * (len(label2id.keys()) - 2) + 2
        # initialize tag
        labels = [[0] * num_labels for i in range(seq_len)]
        if spo_list is not None:
            labels = parse_label(spo_list, label2id, tokens, max_seq_length)

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
            "tok_to_orig_start_index": np.array(tok_to_orig_start_index),
            "tok_to_orig_end_index": np.array(tok_to_orig_end_index),

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


