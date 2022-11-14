# @Time    : 2022/11/14 19:29
# @Author  : tk
# @FileName: t1.py
import numpy as np


a= np.zeros(shape=(1,10))
b= np.zeros(shape=(4,10))

print(np.concatenate([a,b],axis=0).shape)