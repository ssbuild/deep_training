# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/10/9 10:30
import copy
from dataclasses import asdict,dataclass

@dataclass
class _ArgumentsBase:
    # @property
    # def __dict__(self):
    #     return asdict(self)

    def to_dict(self):
        return asdict(self)
    def __deepcopy__(self, memodict={}):
        return self.__class__(**copy.deepcopy(self.to_dict()))
