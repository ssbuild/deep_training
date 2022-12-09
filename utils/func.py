# -*- coding: utf-8 -*-
# @Time    : 2022/11/10 13:46
__all__ = [
  'is_chinese_char'
]

import numpy as np


def is_chinese_char(cp):
  if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
          (cp >= 0x3400 and cp <= 0x4DBF) or  #
          (cp >= 0x20000 and cp <= 0x2A6DF) or  #
          (cp >= 0x2A700 and cp <= 0x2B73F) or  #
          (cp >= 0x2B740 and cp <= 0x2B81F) or  #
          (cp >= 0x2B820 and cp <= 0x2CEAF) or
          (cp >= 0xF900 and cp <= 0xFAFF) or  #
          (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
    return True
  return False

def seq_padding(arr_obj, max_seq_length,dtype=np.int32,pad_val=0):
  pad_len = max_seq_length - len(arr_obj)
  return np.pad(arr_obj, (0, pad_len), 'constant', constant_values=(pad_val, pad_val)) if pad_len > 0 else np.asarray(arr_obj,dtype=dtype)


def seq_pading(array_list, max_seq_length,dtype=np.int64,pad_val=0):
  if max_seq_length is None:
    max_seq_length = max([len(_) for _ in array_list])
  result = []
  for arr in array_list:
    arr = np.asarray(arr,dtype=dtype)
    seqlen = len(arr)
    pad_len = max_seq_length - seqlen
    result.append(np.pad(arr, (0, pad_len), 'constant', constant_values=(pad_val, pad_val)) if pad_len > 0 else arr)
  return result

def seq_pading_with_seqlen(array_list, max_seq_length,dtype=np.int64,pad_val=0):
  if max_seq_length is None:
    max_seq_length = max([len(_) for _ in array_list])
  result = []
  for arr in array_list:
    arr = np.asarray(arr, dtype=dtype)
    seqlen = len(arr)
    pad_len = max_seq_length - seqlen
    result.append((np.pad(arr, (0, pad_len), 'constant', constant_values=(pad_val, pad_val)) if pad_len > 0 else arr, seqlen))
  return result