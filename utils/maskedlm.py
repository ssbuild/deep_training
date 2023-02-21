# @Time    : 2022/11/9 22:29
# @Author  : tk
import numpy as np
import logging
import typing
from transformers import BertTokenizerFast
from .func import is_chinese_char
import copy

def make_mlm_wwm_sample(text : str ,tokenizer,max_seq_length, rng, do_whole_word_mask, max_predictions_per_seq, masked_lm_prob):
    tokenizer: BertTokenizerFast
    # tokenizer, max_seq_length, rng, do_whole_word_mask, max_predictions_per_seq, masked_lm_prob = user_data
    vocab_words = tokenizer.get_vocab()

    o = tokenizer(text, add_special_tokens=True, truncation=True,
                  max_length=max_seq_length,
                  return_token_type_ids=False,
                  return_attention_mask=False)

    input_ids = o['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        if (do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])


    rng.shuffle(cand_indexes)
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(input_ids) * masked_lm_prob))))

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

            masked_lms.append((index, input_ids[index]))

            input_ids[index] = masked_id

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x[0])

    masked_lm_positions = np.zeros(shape=(max_predictions_per_seq,),dtype=np.int64)
    masked_lm_ids = np.zeros(shape=(max_predictions_per_seq,),dtype=np.int64)
    masked_lm_weights = np.zeros(shape=(max_predictions_per_seq,),dtype=np.float32)
    for i,(idx,masked_id) in enumerate(masked_lms):
        masked_lm_positions[i] = idx
        masked_lm_ids[i] = masked_id
        masked_lm_weights[i] = 1

    input_ids = np.asarray(input_ids, dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)


    input_length = np.asarray(len(input_ids), dtype=np.int64)
    pad_len = max_seq_length - input_length
    if pad_len > 0:
        pad_val = tokenizer.pad_token_id
        input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
        attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))

    sample = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'masked_lm_positions': masked_lm_positions,
        'masked_lm_ids': masked_lm_ids,
        'masked_lm_weights': masked_lm_weights,
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