# -*- coding: utf-8 -*-
import json
import os
import sys
sys.path.append('..')


from deep_training.model.nlp.metrics.pointer import metric_for_spo
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from deep_training.model.nlp.models.transformer import TransformerMeta
from pytorch_lightning.callbacks import ModelCheckpoint
import typing
import numpy as np
from deep_training.data_helper import DataHelper
import torch
from pytorch_lightning import Trainer
from deep_training.data_helper import make_dataset_with_args, load_dataset_with_args, \
    load_tokenizer_and_config_with_args
from deep_training.model.nlp.models.casrel import TransformerForHphtlinker, extract_spoes
from transformers import HfArgumentParser, BertTokenizer
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments

train_info_args = {
    'devices': 1,
    'data_backend': 'memory_raw',
    'model_type': 'bert',
    'model_name_or_path':'/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name':'/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name':'/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'do_train': True,
    'do_eval': True,
    #  'train_file': '/data/nlp/nlp_train_data/relation/law/step1_train-fastlabel.json',
    # 'eval_file': '/data/nlp/nlp_train_data/relation/law/step1_train-fastlabel.json',
    # 'label_file': '/data/nlp/nlp_train_data/relation/law/relation_label.json',
    # 'train_file': '/data/nlp/nlp_train_data/myrelation/duie/duie_train.json',
    # 'eval_file': '/data/nlp/nlp_train_data/myrelation/duie/duie_dev.json',
    # 'label_file': '/data/nlp/nlp_train_data/myrelation/duie/duie_schema.json',
    'train_file': '/data/nlp/nlp_train_data/myrelation/re_labels.json',
    'eval_file': '/data/nlp/nlp_train_data/myrelation/re_labels.json',
    'label_file': '/data/nlp/nlp_train_data/myrelation/labels.json',
    'learning_rate': 1e-5,
    'max_epochs': 10,
    'train_batch_size': 10,
    'eval_batch_size': 2,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'train_max_seq_length': 380,
    'eval_max_seq_length': 512,
    'test_max_seq_length': 512,
}

class NN_DataHelper(DataHelper):
    index = -1
    eval_labels = []
    def on_data_ready(self):
        self.index = -1
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        self.index += 1
        tokenizer: BertTokenizer
        tokenizer, max_seq_length, do_lower_case, predicate2id, mode = user_data
        sentence,entities,re_list = data
        spo_list = re_list
        tokens = list(sentence) if not do_lower_case else list(sentence.lower())
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
        input_ids = tokenizer.convert_tokens_to_ids(['CLS'] +tokens + ['SEP'] )
        seqlen = len(input_ids)
        input_ids = np.asarray(input_ids, dtype = np.int64)
        attention_mask = np.asarray([1] * seqlen, dtype=np.int64)
        spoes = {}
        real_label = []
        for s, p, o in spo_list:
            p: int = predicate2id[p]
            real_label.append((s[0], s[1], p, o[0], o[1]))
            s = (s[0] + 1, s[1] + 1)
            o = (o[0] + 1, o[1] + 1)
            if s[1] < max_seq_length - 2 and o[1] < max_seq_length - 2:
                s = (s[0], s[1])
                o = (o[0], o[1], p)
                if s not in spoes:
                    spoes[s] = []
                spoes[s].append(o)

        subject_labels = np.zeros((max_seq_length, 2),dtype=np.float32)
        subject_ids = np.zeros((2,),dtype=np.int64)
        object_labels = np.zeros((max_seq_length, len(predicate2id), 2),dtype=np.float32)

        if spoes:
            for s in spoes:
                subject_labels[s[0], 0] = 1
                subject_labels[s[1], 1] = 1

            for _ in range(1):
                # 随机选一个subject（这里没有实现错误！这就是想要的效果！！）
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids[0] = start
                subject_ids[1] = end
                flag = False
                for o in spoes.get((start,end), []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1
                    flag = True
                if flag:
                    break
        pad_len = max_seq_length - seqlen
        if pad_len > 0:
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(tokenizer.pad_token_id, tokenizer.pad_token_id))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'subject_ids': subject_ids,
            'subject_labels': subject_labels,
            'object_labels': object_labels,
            'seqlen': np.asarray(seqlen,dtype=np.int32)
        }
        if self.index < 5:
            print(tokens)
            print(input_ids[:seqlen])
            # print(subject_labels[:seqlen])
            # print(subject_ids)
            # print(object_labels[:seqlen])

        if mode == 'eval':
            self.eval_labels.append(real_label)

        return d

    #读取标签
    def read_labels_from_file(self,files: typing.List):
        labels = []
        label_filename = files[0]
        with open(label_filename,mode='r',encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                jd = json.loads(line)
                if not jd:
                    continue
                larr = [jd['subject'],jd['predicate'],jd['object']]
                labels.append('+'.join(larr))
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
        return label2id, id2label

    # 读取文件
    def read_data_from_file(self,files: typing.List,mode:str):
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
                        for k,v in entities.items():
                            pts = list(v.values())[0]
                            for pt in pts:
                                entities_label.append((k,pt[0],pt[1]))
                    else:
                        entities_label = None

                    if re_list is not None:
                        re_list_label = []
                        for re_node in re_list:
                            for l,relation in re_node.items():
                                s = relation[0]
                                o = relation[1]
                                re_list_label.append((
                                    # (s['pos'][0], s['pos'][1],s['label']),
                                    # l,
                                    # (o['pos'][0], o['pos'][1],o['label'])
                                    (s['pos'][0], s['pos'][1]),
                                    '+'.join([s['label'],l,o['label']]),
                                    (o['pos'][0], o['pos'][1])
                                ))
                    else:
                        re_list_label = None


                    D.append((jd['text'],entities_label, re_list_label))
        return D


    @staticmethod
    def collate_fn(batch):
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
        max_len = torch.max(o.pop('seqlen'))
        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]
        o['subject_labels'] = o['subject_labels'][:, :max_len]
        # o['subject_ids'] = o['subject_ids']
        o['object_labels'] = o['object_labels'][:, :max_len]

        return o



class MyTransformer(TransformerForHphtlinker, metaclass=TransformerMeta):
    def __init__(self,eval_labels, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.eval_labels = eval_labels
        self.index = 0


    def validation_epoch_end(self, outputs: typing.Union[EPOCH_OUTPUT, typing.List[EPOCH_OUTPUT]]) -> None:
        self.index += 1
        # if self.index <= 1:
        #     self.log('val_f1', 0.0, prog_bar=True)
        #     return

        y_preds, y_trues = [], []
        idx = 0
        for o in outputs:
            logits1, logits2, _, _ = o['outputs']
            output_labels = self.eval_labels[idx * len(logits1):(idx + 1) * len(logits1)]
            idx += 1
            p_spoes = extract_spoes([logits1, logits2])
            t_spoes = output_labels
            y_preds.extend(p_spoes)
            y_trues.extend(t_spoes)

        print(y_preds[:3])
        print(y_trues[:3])
        f1, str_report = metric_for_spo(y_trues, y_preds, self.config.label2id)
        print(f1)
        print(str_report)
        self.log('val_f1', f1, prog_bar=True)

if __name__== '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,data_args)
    save_fn_args = (tokenizer, data_args.max_seq_length,label2id)

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


    dm = load_dataset_with_args(dataHelper, training_args, train_files, eval_files, test_files)

    
    model = MyTransformer(dataHelper.eval_labels,config=config,model_args=model_args,training_args=training_args)
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
        accumulate_grad_batches = training_args.gradient_accumulation_steps,
        num_sanity_val_steps=0,
    )

    if data_args.do_train:
        trainer.fit(model, datamodule=dm)

    if data_args.do_eval:
        trainer.validate(model, datamodule=dm)

    if data_args.do_test:
        trainer.test(model, datamodule=dm)
