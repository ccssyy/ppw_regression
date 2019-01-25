import numpy as np
import h5py
import os

def get_trainset(train_file, item):
    f = h5py.File(train_file)
    trainset = f[item].value
    # f.close()
    # print(trainset)

    return trainset


def get_testset(test_file, item):
    f = h5py.File(test_file)
    testset = np.asarray(f[item].value, dtype=np.float32)
    # f.close()
    # print(testset)

    return testset

