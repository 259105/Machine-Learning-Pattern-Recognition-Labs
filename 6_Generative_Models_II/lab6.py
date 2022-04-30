import numpy
import string
import scipy.special
import itertools
import sys
import data.load as load


def mcol(v):
    return v.reshape((v.size, 1))


if __name__ == "__main__":
    lInf, lPur, lPar = load.load_data()

    lInf_train, lInf_evaluation = load.split_data(lInf, 4)
    lPur_train, lPur_evaluation = load.split_data(lPur, 4)
    lPar_train, lPar_evaluation = load.split_data(lPar, 4)

