import torch
import numpy as np


def get_featuremat(representation, threshold):
    representation = np.matmul(representation, representation.T)
    U, S, Vh = np.linalg.svd(representation, full_matrices=False)
    sval_total = (S ** 2).sum()
    sval_ratio = (S ** 2) / sval_total
    r = np.sum(np.cumsum(sval_ratio) < threshold)
    feature = U[:, 0:r]
    print('-'*40)
    print('Gradient Constraints Summary', feature.shape)
    print('-'*40)

    return feature
