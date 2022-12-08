# -*- coding: utf-8 -*-
import json
import logging
import os
import sys
import typing
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from deep_training.model.nlp.metrics.pointer import metric_for_spo
from deep_training.model.nlp.models.transformer import TransformerMeta
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from pytorch_lightning.callbacks import ModelCheckpoint
from deep_training.data_helper import DataHelper
import numpy as np
import torch
from pytorch_lightning import Trainer
from deep_training.data_helper import make_dataset_with_args, load_dataset_with_args, \
    load_tokenizer_and_config_with_args
from transformers import HfArgumentParser, BertTokenizer
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.model.nlp.models.tplinkerplus import TransformerForTplinkerPlus, extract_spoes, TplinkerArguments

train_info_args = {
    'devices': 1,
    'data_backend': 'memory_raw',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'do_train': True,
    'do_eval': True,
    # 'train_file': '/data/nlp/nlp_train_data/relation/law/step1_train-fastlabel.json',
    # 'eval_file': '/data/nlp/nlp_train_data/relation/law/step1_train-fastlabel.json',
    # 'label_file': '/data/nlp/nlp_train_data/relation/law/relation_label.json',
    # 'train_file': '/data/nlp/nlp_train_data/myrelation/duie/duie_train.json',
    # 'eval_file': '/data/nlp/nlp_train_data/myrelation/duie/duie_dev.json',
    # 'label_file': '/data/nlp/nlp_train_data/myrelation/duie/duie_schema.json',
    'train_file': '/data/nlp/nlp_train_data/myrelation/re_labels.json',
    'eval_file': '/data/nlp/nlp_train_data/myrelation/re_labels.json',
    'label_file': '/data/nlp/nlp_train_data/myrelation/labels.json',
    'learning_rate': 5e-5,
    'max_epochs': 15,
    'train_batch_size': 8,
    'eval_batch_size': 4,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'train_max_seq_length': 200,
    'eval_max_seq_length': 200,
    'test_max_seq_length': 200,
    #tplinker_plus args
    'shaking_type': 'cln_plus', #one of ['cat','cat_plus','cln','cln_plus']
    'inner_enc_type': 'mean_pooling', #one of ['mix_pooling','mean_pooling','max_pooling','lstm']
    'tok_pair_sample_rate': 0,
}


class NN_DataHelper(DataHelper):
    index = -1
    eval_labels = ([],[])

    id2label,label2id = None,None
    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        self.index += 1
        tokenizer: BertTokenizer
        tokenizer, max_seq_length, do_lower_case, label2id, mode = user_data
        sentence, entities, re_list = data
        spo_list = re_list
        tokens = list(sentence) if not do_lower_case else list(sentence.lower())
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
        input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
        seqlen = len(input_ids)
        attention_mask = [1] * seqlen
        input_ids = np.asarray(input_ids, dtype=np.int32)
        attention_mask = np.asarray(attention_mask, dtype=np.int32)


        labels = []
        real_label_e = []
        real_label_re = []
        # for l,s,e in entities:
        #     real_label_e.append((l,s,e))
        #     s += 1
        #     e += 1
        #     if s < max_seq_length - 1 and e < max_seq_length - 1:
        #         k = l + '_EE'
        #         if k not in label2id:
        #             #忽略无关紧要的实体
        #             # logging.warning('entity {} is valid ， it may be not in relation...'.format(l))
        #             continue
        #         labels.append((label2id[k],s,e))

        rel2id = self.task_specific_params['rel2id']
        for s, p, o in spo_list:
            real_label_re.append((s[0], s[1], rel2id[p], o[0], o[1]))
            s = (s[0] + 1, s[1] + 1)
            o = (o[0] + 1, o[1] + 1)
            if s[1] < max_seq_length - 1 and o[1] < max_seq_length - 1:

                labels.append((label2id[p.split('+')[0] + '_EE'], s[0], s[1]))
                labels.append((label2id[p.split('+')[-1] + '_EE'], o[0], o[1]))

                if s[0] <= o[0]:
                    labels.append((label2id[p + '_SH'], s[0], o[0]))
                else:
                    labels.append((label2id[p + '_OH'], o[0],s[0]))

                if s[1] <= o[1]:
                    labels.append((label2id[p + '_ST'], s[1], o[1]))
                else:
                    labels.append((label2id[p + '_OT'], o[1], s[1]))

        labels = np.asarray(labels, dtype=np.int32)
        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'seqlen': np.asarray(max_seq_length,dtype=np.int32),
        }

        if self.index < 5:
            print(tokens)
            print(input_ids[:seqlen])

        if mode == 'eval':
            self.eval_labels[0].append(real_label_e)
            self.eval_labels[1].append(real_label_re)
        return d

    # 读取标签
    def on_get_labels(self, files: typing.List):
        labels = []
        label_filename = files[0]

        entities = set()
        with open(label_filename, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                jd = json.loads(line)
                if not jd:
                    continue
                larr = [jd['subject'], jd['predicate'], jd['object']]
                labels.append('+'.join(larr))
                entities.add(jd['subject'])
                entities.add(jd['object'])

        labels = list(set(labels))
        labels = [i + '_' + j for i in labels for j in ['SH','OH','ST','OT']]
        labels += [i + '_EE' for i in entities]
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}

        NN_DataHelper.label2id = label2id
        NN_DataHelper.id2label = id2label
        return label2id, id2label

    def on_task_specific_params(self):
        labels = list(set([i.rsplit('_', 1)[0] for i in NN_DataHelper.label2id if not i.endswith('_EE')]))
        labels_e = list(set([i.rsplit('_', 1)[0] for i in NN_DataHelper.label2id if i.endswith('_EE')]))
        task_specific_params = {
            'rel2id': {l: i for i, l in enumerate(labels)},
            'id2rel': {i: l for i, l in enumerate(labels)},
            'ent2id': {l: i for i, l in enumerate(labels_e)},
            'id2ent': {i: l for i, l in enumerate(labels_e)},
            'label2id': NN_DataHelper.label2id,
            'id2label': NN_DataHelper.id2label,
        }
        self.task_specific_params = task_specific_params
        return task_specific_params

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
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
                            pts = [_ for a_ in list(v.values()) for _ in a_]
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
        return D[0:100]



    # @staticmethod
    # def collate_fn(batch):
    #     return batch

    # batch dataset
    @staticmethod
    def batch_transform(batch):
        bs = len(batch)
        o = {}
        labels_info = []
        for i, b in enumerate(batch):
            labels_info.append(b.pop('labels', []))
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])
        max_len = torch.max(o.pop('seqlen'))

        shaking_len = int(max_len * (max_len + 1) / 2)
        labels = torch.zeros(size=(bs ,len(NN_DataHelper.label2id),shaking_len), dtype=torch.long)
        get_pos = lambda x0, x1: x0 * max_len + int(x1 - x0 * (x0 + 1) / 2)

        for linfo, label in zip(labels_info, labels):
            for l, s, e in linfo:
                assert s <= e
                if s >= max_len - 1 or e >= max_len - 1:
                    continue
                label[l][get_pos(s, e)] = 1


        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        o['labels'] = labels
        return o


class MyTransformer(TransformerForTplinkerPlus, metaclass=TransformerMeta):
    def __init__(self,eval_labels,*args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.index = 0
        self.eval_labels = eval_labels
        self.rel2id = self.config.task_specific_params['rel2id']
        self.id2rel = self.config.task_specific_params['id2rel']



    def validation_epoch_end(self, outputs: typing.Union[EPOCH_OUTPUT, typing.List[EPOCH_OUTPUT]]) -> None:
        self.index += 1
        # if self.index < 2:
        #     self.log('val_f1', 0.0, prog_bar=True)
        #     return
        threshold = 1e-8
        # 关系标注
        eval_labels = self.eval_labels[1]
        y_preds, y_trues = [], []
        idx = 0
        for o in outputs:
            logits, _ = o['outputs']
            output_labels = eval_labels[idx*len(logits):(idx + 1)*len(logits)]
            idx += 1
            p_spoes = extract_spoes(logits,self.config.id2label,self.rel2id,threshold)
            t_spoes = output_labels
            y_preds.extend(p_spoes)
            y_trues.extend(t_spoes)

        print(y_preds[:3])
        print(y_trues[:3])
        f1, str_report = metric_for_spo(y_trues, y_preds, self.rel2id)
        print(f1)
        print(str_report)
        self.log('val_f1', f1, prog_bar=True)





if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments,TplinkerArguments))
    model_args, training_args, data_args , tplinker_args = parser.parse_dict(train_info_args)

    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,
                                                                                data_args)
    token_fn_args_dict = {
        'train': (tokenizer, data_args.train_max_seq_length, model_args.do_lower_case, label2id, 'train'),
        'eval': (tokenizer, data_args.eval_max_seq_length, model_args.do_lower_case, label2id, 'eval'),
        'test': (tokenizer, data_args.test_max_seq_length, model_args.do_lower_case, label2id, 'test')
    }

    N = 1
    train_files, eval_files, test_files = [], [], []
    for i in range(N):
        intermediate_name = data_args.intermediate_name + '_{}'.format(i)
        if data_args.do_train:
            train_files.append(
                make_dataset_with_args(dataHelper, data_args.train_file, token_fn_args_dict['train'], data_args,
                                       intermediate_name=intermediate_name, shuffle=True, mode='train'))
        if data_args.do_eval:
            eval_files.append(
                make_dataset_with_args(dataHelper, data_args.eval_file, token_fn_args_dict['eval'], data_args,
                                       intermediate_name=intermediate_name, shuffle=False, mode='eval'))
        if data_args.do_test:
            test_files.append(
                make_dataset_with_args(dataHelper, data_args.test_file, token_fn_args_dict['test'], data_args,
                                       intermediate_name=intermediate_name, shuffle=False, mode='test'))

    dm = load_dataset_with_args(dataHelper, training_args,train_files,eval_files, test_files)

    model = MyTransformer(dataHelper.eval_labels,tplinker_args=tplinker_args, config=config, model_args=model_args, training_args=training_args)
    checkpoint_callback = ModelCheckpoint(monitor="val_f1", save_last=True, every_n_epochs=1)
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,  
        enable_progress_bar=True,
        default_root_dir=data_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        num_sanity_val_steps=0,
    )

    if data_args.do_train:
        trainer.fit(model, datamodule=dm)

    if data_args.do_eval:
        trainer.validate(model, datamodule=dm)

    if data_args.do_test:
        trainer.test(model, datamodule=dm)