# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 13:31
import os
import json
import typing
import numpy as np
import torch
from asmodels.data_helper import DataHelper
from transformers import BertTokenizer #仅编写提示使用，实际不一定使用

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
        seqlen = len(input_ids)
        attention_mask = [1] * seqlen

        input_ids = np.asarray(input_ids, dtype=np.int64)
        attention_mask = np.asarray(attention_mask, dtype=np.int64)


        num_labels = len(predicate2id)
        labels = np.zeros(shape=(seqlen - 2, num_labels))
        if spo_list is not None:
            for s,p,o in spo_list:
                if s[1] > seqlen - 2 or o[1] > seqlen - 2:
                    continue

                s_ids = [s[0], s[1]]
                o_ids = [o[0], o[1]]

                label_for_s = predicate2id[p]
                label_for_o = label_for_s + ((len(predicate2id.keys()) - 2) // 2)

                slen = s_ids[1] - s_ids[0] + 1
                labels[s[0]][label_for_s] = 1
                for i in range(slen - 1):
                    labels[s[0] + i + 1][1] = 1

                labels[o[0]][label_for_o] = 1
                olen = o_ids[1] - o_ids[0] + 1
                print(o_ids,olen)
                for i in range(olen - 1):
                    labels[o[0] + i + 1][1] = 1

            # if token wasn't assigned as any "B"/"I" tag, give it an "O" tag for outside
            for i in range(seqlen):
                if labels[i] == [0] * num_labels:
                    labels[i][0] = 1


        edge = np.expand_dims(np.asarray([[1] + [0] * (num_labels - 1)]),axis=1)
        labels = np.concatenate([edge,labels,edge],axis=1)

        pad_len = max_seq_length - len(input_ids)
        if pad_len:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            labels = np.concatenate([labels, np.zeros(pad_val,num_labels)],axis=0)


        mask = np.logical_and(input_ids != tokenizer.pad_token_id,
                              np.logical_and(input_ids != tokenizer.cls_token_id, input_ids != tokenizer.sep_token_id))

        d = {
            "input_ids": np.array(input_ids,dtype=np.int64),
            "attention_mask": np.asarray(attention_mask,dtype=np.int64),
            "mask": np.asarray(mask,dtype=np.int64),
            "labels": np.array(labels,dtype=np.int64),
            "seqlen": np.array(seqlen,dtype=np.int64),
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
        labels = ['O','I'] + [tag  + l for tag in ['','unused'] for l in labels ]
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

        o['labels'] = o['labels'][:, :max_len]
        return o


