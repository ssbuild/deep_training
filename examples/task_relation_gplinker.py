# -*- coding: utf-8 -*-
import json
import os
import sys
import typing

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from pytorch_lightning.callbacks import ModelCheckpoint
from deep_training.data_helper import DataHelper
import numpy as np
import torch
from pytorch_lightning import Trainer
from deep_training.data_helper import make_all_dataset_with_args, load_all_dataset_with_args, \
    load_tokenizer_and_config_with_args
from transformers import HfArgumentParser, BertTokenizer
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.model.nlp.models.gplinker import TransformerForGplinker, extract_spoes, extract_spoes_from_labels

train_info_args = {
    'devices': 1,
    'data_backend': 'memory_raw',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'do_train': True,
    'do_eval': True,
    'train_file': '/data/nlp/nlp_train_data/relation/law/step1_train-fastlabel.json',
    'eval_file': '/data/nlp/nlp_train_data/relation/law/step1_train-fastlabel.json',
    'label_file': '/data/nlp/nlp_train_data/relation/law/relation_label.json',
    'learning_rate': 5e-5,
    'learning_rate_for_task': 3e-4,
    'max_epochs': 15,
    'train_batch_size': 32,
    'eval_batch_size': 8,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length': 160,
}


def convert_feature(data: typing.Any, user_data: tuple):
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
    input_ids = np.asarray(input_ids, dtype=np.int32)
    attention_mask = np.asarray(attention_mask, dtype=np.int32)


    max_target_len = 60
    entity_labels = np.zeros(shape=(2, max_target_len, 2), dtype=np.int32)
    head_labels = np.zeros(shape=(len(predicate2id), max_target_len, 2), dtype=np.int32)
    tail_labels = np.zeros(shape=(len(predicate2id), max_target_len, 2), dtype=np.int32)

    entity_labels_tmp = [set() for _ in range(2)]
    head_labels_tmp = [set() for _ in range(len(predicate2id))]
    tail_labels_tmp = [set() for _ in range(len(predicate2id))]
    for s, p, o in spo_list:
        s = (s[0] + 1, s[1] + 1)
        o = (o[0] + 1, o[1] + 1)
        if s[1] < max_seq_length - 2 and o[1] < max_seq_length - 2:
            entity_labels_tmp[0].add((s[0], s[1]))
            entity_labels_tmp[1].add((o[0], o[1]))
            p: int = predicate2id[p]
            head_labels_tmp[p].add((s[0], s[1]))
            tail_labels_tmp[p].add((o[0], o[1]))

    def feed_label(x, pts_list):
        tlen = 0
        for p,pts in enumerate(pts_list):
            for seq,pos in enumerate(pts):
                tlen += 1
                x[p][seq][0] = pos[0]
                x[p][seq][1] = pos[1]
        return tlen

    targetlen1 = feed_label(entity_labels, list(map(lambda x: list(x), entity_labels_tmp)))
    targetlen2 = feed_label(head_labels, list(map(lambda x: list(x), head_labels_tmp)))
    targetlen3 = feed_label(tail_labels, list(map(lambda x: list(x), tail_labels_tmp)))

    targetlen = np.asarray(np.max([targetlen1,targetlen2,targetlen3,1]),dtype=np.int32)
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
        'targetlen': targetlen,
    }
    print(seqlen,targetlen)
    return d


class NN_DataHelper(DataHelper):
    index = 0

    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        return convert_feature(data, user_data)

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

        targetlen = o.pop('targetlen')
        max_tarlen = torch.max(targetlen)

        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        o['entity_labels'] = o['entity_labels'][:, :, :max_tarlen]
        o['head_labels'] = o['head_labels'][:, :, :max_tarlen]
        o['tail_labels'] = o['tail_labels'][:, :, :max_tarlen]
        return o


class MyTransformer(TransformerForGplinker):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.index = 0

    def validation_epoch_end(self, outputs: typing.Union[EPOCH_OUTPUT, typing.List[EPOCH_OUTPUT]]) -> None:
        self.index += 1
        if self.index <= 2:
            self.log('val_f1', 0.0)
            return


        threshold = 0

        y_preds, y_trues = [], []
        for o in outputs:
            logits1, logits2, logits3, entity_labels, head_labels, tail_labels = o['outputs']
            for p1, p2, p3, l1, l2, l3 in zip(logits1, logits2, logits3, entity_labels, head_labels, tail_labels):
                p_spoes = extract_spoes([p1, p2, p3], threshold=threshold)
                t_spoes = extract_spoes_from_labels([l1, l2, l3])
                y_preds.append(p_spoes)
                y_trues.append(t_spoes)

        print(y_preds[:10])
        print(y_trues[:10])
        # print(f1)
        # print(str_report)
        # self.log('val_f1', f1, prog_bar=True)
        self.log('val_f1', 0)


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,
                                                                                data_args)
    save_fn_args = (tokenizer, data_args.max_seq_length, label2id)

    N = 1
    train_files, eval_files, test_files = [], [], []
    for i in range(N):
        intermediate_name = data_args.intermediate_name + '_{}'.format(i)
        train_file, eval_file, test_file = make_all_dataset_with_args(dataHelper, save_fn_args, data_args,
                                                                      intermediate_name=intermediate_name,
                                                                      num_process_worker=0)
        train_files.append(train_file)
        eval_files.append(eval_file)
        test_files.append(test_file)

    dm = load_all_dataset_with_args(dataHelper, training_args, train_files, eval_files, test_files)
    model = MyTransformer(with_efficient=False, config=config, model_args=model_args, training_args=training_args)
    checkpoint_callback = ModelCheckpoint(monitor="val_f1", save_last=True, every_n_epochs=1)
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,  # limiting got iPython runs
        enable_progress_bar=True,
        default_root_dir=data_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps
    )

    if data_args.do_train:
        trainer.fit(model, datamodule=dm)

    if data_args.do_eval:
        trainer.validate(model, datamodule=dm)

    if data_args.do_test:
        trainer.test(model, datamodule=dm)
