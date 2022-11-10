# @Time    : 2022/11/6 10:40
import typing
import numpy as np
from fastdatasets.utils.NumpyAdapter import ParallelNumpyWriter,NumpyReaderAdapter
from transformers import BertTokenizer
from fastdatasets.record import RECORD

# 切分词
def tokenize_data(data_index: int, data: typing.Any, user_data: tuple):
    tokenizer: BertTokenizer
    max_seq_length = user_data

    a : np.ndarray = np.random.random(size=(10,512,512))

    node = {
        'input_ids': np.asarray(a.tobytes()),
        'attention_mask': np.random.randint(0,21128,size=(1000)),
    }
    return node


def make_dataset(data,data_backend,outputfile):
    parallel_writer = ParallelNumpyWriter(num_process_worker=8,input_queue_size=100,output_queue_size=100)
    parallel_writer.initailize_input_hook(tokenize_data, (64,))
    parallel_writer.initialize_writer(outputfile,data_backend,options=RECORD.TFRecordOptions(compression_type=None))

    parallel_writer.parallel_apply(data)


def test(data,data_backend,outputfile):
    make_dataset(data,data_backend,outputfile)
    dataset = NumpyReaderAdapter.load(outputfile, data_backend,options=RECORD.TFRecordOptions(compression_type=None))
    if isinstance(dataset, typing.Iterator):
        for d in dataset:
            print(d)
            break
    else:
        for i in range(len(dataset)):
            print(dataset[i])
            break
        print('total count', len(dataset))
if __name__ == '__main__':
    data = [0] * 1000
    test(data,'record','./data.record')
    # test(data,'leveldb', './data.leveldb')
    # test(data,'lmdb', './data.lmdb')