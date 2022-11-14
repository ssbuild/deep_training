# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 13:31
import os
import json
import typing
import numpy as np
import torch
from asmodels.data_helper import DataHelper
from transformers import BertTokenizer




class NN_DataHelper(DataHelper):
    index = 0
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
        seqlen = len(input_ids)
        entity_labels = np.zeros(shape=(2,max_seq_length,max_seq_length))
        head_labels = np.zeros(shape=(len(predicate2id),max_seq_length,max_seq_length))
        tail_labels = np.zeros(shape=(len(predicate2id),max_seq_length,max_seq_length))

        entity_labels_tmp = [set() for _ in range(2)]
        head_labels_tmp = [set() for _ in range(len(predicate2id))]
        tail_labels_tmp = [set() for _ in range(len(predicate2id))]
        for s, p, o in spo_list:
            if s[1] < max_seq_length - 2 and o[1] < max_seq_length - 2:
                entity_labels_tmp[0].add((s[0], s[1]))
                entity_labels_tmp[1].add((o[0], o[1]))
                p:int = predicate2id[p]
                head_labels_tmp[p].add((s[0], s[1]))
                tail_labels_tmp[p].add((o[0], o[1]))

        x1 = list(map(lambda x: list(x), entity_labels_tmp))
        x2 = list(map(lambda x: list(x), head_labels_tmp))
        x3 = list(map(lambda x: list(x), tail_labels_tmp))
        def feed_label(x,pts_list):
            for i,pts in enumerate(pts_list):
                for p in pts:
                    x[i][p[0]][p[1]] = 1
        feed_label(entity_labels,x1)
        feed_label(head_labels, x2)
        feed_label(tail_labels, x3)

        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'entity_labels': entity_labels,
            'head_labels': head_labels,
            'tail_labels': tail_labels,
            'seqlen': seqlen,
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

        o['entity_labels'] = o['entity_labels'][:,:, :max_len,:max_len]
        o['head_labels'] = o['head_labels'][:, :, :max_len,:max_len]
        o['tail_labels'] = o['tail_labels'][:, :, :max_len,:max_len]
        return o

