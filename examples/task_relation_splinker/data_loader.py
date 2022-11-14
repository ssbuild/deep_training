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
    labels = np.zeros(shape=(max_length,num_labels))

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



class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):

        tokenizer: BertTokenizer
        tokenizer, max_seq_length, predicate2id, mode = user_data
        sentence, entities, re_list = data
        spo_list = re_list
        tokens = list(sentence)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
        input_ids = tokenizer.convert_tokens_to_ids(['CLS'] + tokens + ['SEP'])
        input_length = len(input_ids)
        attention_mask = [1] * input_length

        input_ids = np.asarray(input_ids, dtype=np.int64)
        attention_mask = np.asarray(attention_mask, dtype=np.int64)

        seq_len = len(input_ids)
        # 2 tags for each predicate + I tag + O tag
        num_labels = 2 * (len(predicate2id.keys()) - 2) + 2
        # initialize tag
        labels = [[0] * num_labels for _ in range(seq_len)]
        if spo_list is not None:
            labels = parse_label(spo_list, predicate2id, tokens, max_seq_length)

        tok_to_orig_start_index = []
        tok_to_orig_end_index = []
        orig_to_tok_index = []
        if seq_len > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            tok_to_orig_start_index = tok_to_orig_start_index[0:(max_seq_length - 2)]
            tok_to_orig_end_index = tok_to_orig_end_index[0:(max_seq_length - 2)]


        # "O" tag for [PAD], [CLS], [SEP] token
        outside_label = [[1] + [0] * (num_labels - 1)]

        labels = outside_label + labels + outside_label
        tok_to_orig_start_index = [-1] + tok_to_orig_start_index + [-1]
        tok_to_orig_end_index = [-1] + tok_to_orig_end_index + [-1]

        pad_len = max_seq_length - len(input_ids)
        if pad_len:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            labels = labels + outside_label * (max_seq_length - len(labels))
            tok_to_orig_start_index = tok_to_orig_start_index + [-1] * (max_seq_length - len(tok_to_orig_start_index))
            tok_to_orig_end_index = tok_to_orig_end_index + [-1] * (max_seq_length - len(tok_to_orig_end_index))


        mask = np.logical_and(input_ids != tokenizer.pad_token_id,
                              np.logical_and(input_ids != tokenizer.cls_token_id, input_ids != tokenizer.sep_token_id))

        d = {
            "input_ids": np.array(input_ids),
            "attention_mask": np.asarray(attention_mask),
            "mask": np.asarray(mask),
            "labels": np.array(labels),
            "seq_len": np.array(seq_len),
            # "tok_to_orig_start_index": np.array(tok_to_orig_start_index),
            # "tok_to_orig_end_index": np.array(tok_to_orig_end_index),

        }
        return d

        # 读取标签

    @staticmethod
    def read_labels_from_file(files: typing.List):
        labels = []
        label_filename = files[0]
        with open(label_filename, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                jd = json.loads(line)
                if not jd:
                    continue
                larr = [jd['subject'], jd['predicate'], jd['object']]
                labels.append('+'.join(larr))
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
        return label2id, id2label

    # 读取文件
    @staticmethod
    def read_data_from_file(files: typing.List, mode: str):
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    jd = json.loads(line)
                    if not jd:
                        continue

                    entities = jd.get('entities', None)
                    re_list = jd.get('re_list', None)

                    if entities:
                        entities_label = []
                        for k, v in entities.items():
                            pts = list(v.values())[0]
                            for pt in pts:
                                entities_label.append((k, pt[0], pt[1]))
                    else:
                        entities_label = None

                    if re_list is not None:
                        re_list_label = []
                        for re_node in re_list:
                            for l, relation in re_node.items():
                                s = relation[0]
                                o = relation[1]
                                re_list_label.append((
                                    # (s['pos'][0], s['pos'][1],s['label']),
                                    # l,
                                    # (o['pos'][0], o['pos'][1],o['label'])
                                    (s['pos'][0], s['pos'][1]),
                                    '+'.join([s['label'], l, o['label']]),
                                    (o['pos'][0], o['pos'][1])
                                ))
                    else:
                        re_list_label = None

                    D.append((jd['text'], entities_label, re_list_label))
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


