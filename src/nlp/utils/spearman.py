# -*- coding: utf-8 -*-
# @Time    : 2022/12/21 17:13
import numpy as np
import scipy
from scipy.stats import stats
from sklearn.metrics.pairwise import paired_distances


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    norms = (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def evaluate_spearman(a_vecs,b_vecs,labels):
    sims = 1 - paired_distances(a_vecs,b_vecs,metric='cosine')
    correlation,_  = stats.spearmanr(labels,sims)
    return correlation
