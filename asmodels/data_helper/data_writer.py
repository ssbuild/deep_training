# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 11:02
import typing
from fastdatasets.utils import NumpyWriterAdapter,ParallelNumpyWriter

__all__ = [
    'DataWriteHelper',
    'NumpyWriterAdapter',
    'ParallelNumpyWriter'
]

class DataWriteHelper:
    def __init__(self,
                 input_fn: typing.Callable[
                     [int, typing.Any, tuple], typing.Union[typing.Dict, typing.List, typing.Tuple]],
                 input_fn_args:typing.Tuple,
                 outfile:str,
                 backend='record',
                 num_process_worker=8,
                 shuffle=True):
        assert backend in ['record', 'lmdb', 'leveldb']

        self.input_fn = input_fn
        self.input_fn_args = input_fn_args
        assert isinstance(input_fn_args, tuple)
        self.outfile = outfile
        self._backend_type = backend
        assert num_process_worker > 0
        self._parallel_writer = ParallelNumpyWriter(num_process_worker=num_process_worker,shuffle=shuffle)

    @property
    def backend_type(self):
        return self._backend_type

    @backend_type.setter
    def backend_type(self, value):
        self._backend_type = value

    # 多进程写大文件
    def save(self,data: list):
        self._parallel_writer.initailize_input_hook(self.input_fn, self.input_fn_args)
        self._parallel_writer.initialize_writer(self.outfile ,self.backend_type)
        self._parallel_writer.parallel_apply(data)
