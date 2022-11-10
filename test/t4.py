# @Time    : 2022/11/10 23:23
# @Author  : tk
# @FileName: t4.py
import numpy as np
from scipy.sparse import coo_matrix

_row  = np.array([0, 3, 1, 0])
_col  = np.array([0, 3, 1, 2])
_data = np.array([4, 5, 7, 9])
coo = coo_matrix((_data, (_row, _col)), shape=(4, 4), dtype=np.int32)


print(print(type(coo.todense())))

print(coo.toarray() )
