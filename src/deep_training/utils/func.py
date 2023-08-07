# -*- coding: utf-8 -*-
# @Time    : 2022/11/10 13:46
import numpy as np
from .language import is_chinese_char

__all__ = [
  'is_chinese_char'
]

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