# @Time    : 2022/11/10 23:03
# @Author  : tk
# @FileName: t1.py
#
import data_serialize
import numpy as np


def test_numpyobject():
    a = np.random.randint(0, 21128, size=(10,), dtype=np.int64)
    b = np.random.rand(10,512, 512)
    c = np.asarray(b'The china')

    val1 = data_serialize.NumpyObject(
        header='',
        dtype=str(a.dtype),
        shape=list(a.shape),
        int64=a.reshape((-1,)).tolist(),
    )
    val2 = data_serialize.NumpyObject(
        header='',
        dtype=str(b.dtype),
        shape=list(b.shape),
        float64=b.reshape((-1,)).tolist(),
    )
    val3 = data_serialize.NumpyObject(
        header='',
        dtype=str(c.dtype),
        shape=list(c.shape),
        bytes=c.tobytes(),
    )

    example = data_serialize.NumpyObjectMap(numpyobjects={
        "item_0": val1,
        "item_1": val2,
        "item_2": val3}
    )
    # 序列化
    serialize = example.SerializeToString()
    print(len(serialize))
    # print(serialize)
    #
    # # 反序列化
    # example = data_serialize.NumpyObjectMap()
    # example.ParseFromString(serialize)
    # print(example)



test_numpyobject()