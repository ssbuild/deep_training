# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 11:02

import typing
from fastdatasets.utils.numpyadapter import NumpyWriterAdapter,ParallelNumpyWriter

__all__ = [
    'DataWriteHelper',
    'NumpyWriterAdapter',
    'ParallelNumpyWriter'
]

class DataWriteHelper:
    def __init__(self,
                 input_fn: typing.Callable[
                     [typing.Any, tuple], typing.Union[typing.Dict, typing.List, typing.Tuple]],
                 input_fn_args: typing.Union[typing.Tuple,typing.Dict],
                 outfile: typing.Union[str,list],
                 backend='record',
                 num_process_worker=0,
                 shuffle=True):
        assert backend in ['record', 'lmdb', 'leveldb','memory','memory_raw']

        self.input_fn = input_fn
        self.input_fn_args = input_fn_args
        self.outfile = outfile
        self._backend_type = backend
        self._parallel_writer = ParallelNumpyWriter(num_process_worker=num_process_worker,shuffle=shuffle)

    @property
    def backend_type(self):
        return self._backend_type

    @backend_type.setter
    def backend_type(self, value):
        self._backend_type = value

    # 多进程写大文件
    def save(self,data: list):
        self._parallel_writer.open(self.outfile ,self.backend_type)
        self._parallel_writer.write(data,self.input_fn, self.input_fn_args)