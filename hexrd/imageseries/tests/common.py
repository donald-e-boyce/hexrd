import numpy as np
import unittest

from hexrd import imageseries

_NFXY = (3, 7, 5)

class ImageSeriesTest(unittest.TestCase):
    pass

def make_array():
    a = np.zeros(_NFXY)
    ind = np.array([0,1,2])
    a[ind, 1,2] = 1 + ind
    return a

def make_array_ims():
    is_a = imageseries.open(None, 'array', data=make_array(),
                            meta=make_meta())
    return is_a

def compare(ims1, ims2):
    """compare two imageseries"""
    if len(ims1) != len(ims2):
        raise ValueError("lengths do not match")

    if ims1.dtype is not ims2.dtype:
        raise ValueError("types do not match")

    maxdiff = 0.0
    for i in range(len(ims1)):
        f1 = ims1[i]
        f2 = ims2[i]
        fdiff = np.linalg.norm(f1 - f2)
        maxdiff = np.maximum(maxdiff, fdiff)

    return maxdiff

def make_meta():
    return {'testing': '1,2,3'}

def compare_meta(ims1, ims2):
    # check metadata
    # . originally only checked for mutable items

    m1 = ims1.metadata
    m2 = ims2.metadata

    if not len(m1) == len(m2):
        return False

    for k in m1:
        if k not in m2:
            return False
        if type(m1[k]) is not type(m2[k]):
            return False
        if not m1[k] == m2[k]:
            return False

    return True
