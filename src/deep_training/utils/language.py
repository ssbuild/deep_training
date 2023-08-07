# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/7 13:52

__all__ = [
  'is_chinese_char'
]

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