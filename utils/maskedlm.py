# @Time    : 2022/11/9 22:29
# @Author  : tk
# @FileName: maskedlm.py
import numpy as np
import logging
import typing
from transformers import BertTokenizerFast
from .func import is_chinese_char
import copy
import jieba
jieba.setLogLevel(log_level=logging.WARNING)

#tokens does not contail [CLS] [SEP]
def get_wwm_cand_indexes(text:str,tokens:typing.List[str],offset_mapping:typing.List[tuple]):
    arr_fenci = jieba.lcut(text)
    offsets_fenci = []
    pos = 0
    for fenci in arr_fenci:
        ll = len(fenci)
        offsets_fenci.append((pos, pos + ll))
        pos += ll
    offsets_fenci= sorted(offsets_fenci)
    offset_mapping = sorted(offset_mapping)
    cand_ids = []
    pos_i = 0
    while len(offsets_fenci):
        offset = offsets_fenci.pop(0)
        index_flag = None
        for i in range(pos_i,len(offset_mapping)):
            offset2 = offset_mapping[i]
            if offset2[0] >= offset[1]:
                index_flag = i
                pos_i = i + 1
                break
        if index_flag is not None:
            cand_ids.append(index_flag)
    cand_indexes = []
    pos = 0
    for i in range(len(cand_ids)):
        cand_indexes.append(list(range(pos,cand_ids[i])))
        pos = cand_ids[i]
    tmp = []
    for pos in cand_indexes:
        tmp.extend(pos)
    tmp = set(tmp)
    for i,(token,offset) in enumerate(zip(tokens,offset_mapping)):
        if i in tmp:
            continue
        if token.startswith('##') :
            if len(token[2:]) == 1 and is_chinese_char(ord(token[2])):
                cand_indexes.append([i])
            else:
                cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
    # tmp = set()
    # length = 0
    # for idexes in cand_indexes:
    #     length += len(idexes)
    #     for index in idexes:
    #         tmp.add(index)
    # assert len(tmp) == length
    return cand_indexes


def make_mlm_wwm_sample(text : str ,tokenizer,max_seq_length, rng, do_whole_word_mask, max_predictions_per_seq, masked_lm_prob):
    tokenizer: BertTokenizerFast
    # tokenizer, max_seq_length, rng, do_whole_word_mask, max_predictions_per_seq, masked_lm_prob = user_data
    vocab_words = tokenizer.get_vocab()

    o = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_seq_length - 2,
                  return_offsets_mapping=True)
    input_ids = o['input_ids']
    offset_mapping = o['offset_mapping']
    if do_whole_word_mask:
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        cand_indexes = get_wwm_cand_indexes(text, tokens, offset_mapping)
    else:
        cand_indexes = [(i, i + 1) for i in input_ids]

    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    cand_indexes = [[index + 1 for index in indexes] for indexes in cand_indexes]
    rng.shuffle(cand_indexes)
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(input_ids) * masked_lm_prob))))
    labels = copy.deepcopy(input_ids)

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_id = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_id = tokenizer.mask_token_id
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_id = input_ids[index]
                # 10% of the time, replace with random word
                else:
                    masked_id = rng.randint(0, len(vocab_words) - 1)

            input_ids[index] = masked_id

            masked_lms.append((index, masked_id))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x[0])

    input_ids = np.asarray(input_ids, dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    weight = np.zeros_like(input_ids, dtype=np.float32)
    for index, _ in masked_lms:
        weight[index] = 1.0

    input_length = np.asarray(len(input_ids), dtype=np.int64)
    pad_len = max_seq_length - input_length
    if pad_len > 0:
        pad_val = tokenizer.pad_token_id
        input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
        attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
        labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(0, 0))
        weight = np.pad(weight, (0, pad_len), 'constant', constant_values=(0, 0))
    sample = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'weight': weight,
        'seqlen': input_length
    }
    return sample


# 切分词
def make_gpt2_sample(data: typing.Any, user_data: tuple):
    tokenizer, max_seq_length = user_data

    x = data
    if isinstance(x, tuple):
        o = tokenizer.encode_plus(text=x[0], text_pair=x[1], max_length=max_seq_length, truncation=True,
                                  add_special_tokens=True)
    else:
        o = tokenizer.encode_plus(x, max_length=max_seq_length, truncation=True, add_special_tokens=True, )

    input_ids = np.asarray(o['input_ids'], dtype=np.int64)
    attention_mask = np.asarray(o['attention_mask'], dtype=np.int64)
    token_type_ids = np.asarray(o['token_type_ids'], dtype=np.int64)

    seqlen = np.asarray(len(input_ids), dtype=np.int64)
    pad_len = max_seq_length - len(input_ids)
    if pad_len > 0:
        input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(0, 0))
        attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
        token_type_ids = np.pad(token_type_ids, (0, pad_len), 'constant', constant_values=(0, 0))
    d = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'labels': input_ids,
        'seqlen': seqlen
    }
    return d