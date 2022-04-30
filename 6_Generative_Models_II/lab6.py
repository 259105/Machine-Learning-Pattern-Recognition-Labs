import numpy
import string
import scipy.special
import itertools
import sys
import data.load as load


def mcol(v):
    return v.reshape((v.size, 1))


def load_data():
    lInf = []

    f = open('data/inferno.txt', encoding="ISO-8859-1")

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    f = open('data/purgatorio.txt', encoding="ISO-8859-1")

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    f = open('data/paradiso.txt', encoding="ISO-8859-1")

    for line in f:
        lPar.append(line.strip())
    f.close()

    return lInf, lPur, lPar


def split_data(lines, n):
    lTrain, lTest = [], []
    for i in range(len(lines)):
        if i % n == 0:
            lTest.append(lines[i])
        else:
            lTrain.append(lines[i])

    return lTrain, lTest


## My solution 1 - Dictionaries of frequencies ###

def S1_estimateModelMy(hlTraining, eps=0.1):
    """
    Build frequency dictionaries for each class.

    hlTercets: dict whose keys are the classes, and the values are the list of tercets of each class.
    eps: smoothing factor (pseudo-count)

    Return: dictionary h_clsLogProb whose keys are the classes. For each class, h_clsLogProb[cls] is a dictionary whose
    keys are words and values are the corresponding log-frequencies (model parameters for class cls)
    """

    # Class dictionary
    clsDic = set()
    # temp glob set in which are saved all the distinct word
    globalDic = set()
    # For each class
    for cls in hlTraining:
        clsDic.add(cls)
        # For each line
        for line in hlTraining[cls]:
            words = line.split()
            # For each word
            for word in words:
                globalDic.add(word)
    # dictionary of (distWord, i)
    globalDic = {word: i for (i, word) in enumerate(globalDic)}
    # dictionary of (class, i)
    clsDic = {cls: i for (i, cls) in enumerate(sorted(clsDic))}
    print(clsDic)

    # matrix of shape (NumClasses, NumWords), fill the matrix by eps [hyperParamiter] to avoid problems
    clsLogProb = numpy.full((len(hlTraining), len(globalDic)), eps)

    # Estimate counts
    # Loop over class labels
    for cls in hlTraining:
        # Loop over lines of that class
        for line in hlTraining[cls]:
            words = line.split()
            # Loop over the words of that line
            for word in words:
                clsLogProb[clsDic[cls], globalDic[word]] += 1

    # Get all occurrences of words in cls and sum them. This is the number of words (including pseudo-counts)
    logNWordsCls = numpy.log(mcol(clsLogProb.sum(1)))  # (1, NumClasses) for each the sum of words within
    clsLogProb = numpy.log(clsLogProb)
    # compute the logFrequency
    return {
        "clsLogProb": clsLogProb - logNWordsCls,
        "clsDic": clsDic,
        "globalDic": globalDic
    }


def S1_compute_logLikelihoodsMy(model, text):
    """
    Compute the array of log-likelihoods for each class for the given text
    h_clsLogProb is the dictionary of model parameters as returned by S1_estimateModel
    The function returns a dictionary of class-conditional log-likelihoods
    """
    # Create the logLikelihood array (1, NumClasses)
    logLikelihoodCls = numpy.zeros((model["clsLogProb"].shape[0], 1))
    # Loop over classes
    for cls in range(model["clsLogProb"].shape[0]):
        # Loop over words
        for word in text.split():
            if word in model["globalDic"]:
                # Sums the logProb of words that are in the global dictionary
                # this is done for each class
                logLikelihoodCls[cls] += model["clsLogProb"][cls, model["globalDic"][word]]
    return logLikelihoodCls


def S1_compute_logLikelihoodMatrixMy(model, lTercets, hCls2Idx=None):
    """
    Compute the matrix of class-conditional log-likelihoods for each class each tercet in lTercets

    h_clsLogProb is the dictionary of model parameters as returned by S1_estimateModel
    lTercets is a list of tercets (list of strings)
    hCls2Idx: map between textual labels (keys of h_clsLogProb) and matrix rows. If not provided, automatic mapping based on alphabetical oreder is used

    Returns a #cls x #tercets matrix. Each row corresponds to a class.
    """
    if hCls2Idx is None:
        hCls2Idx = {cls: idx for idx, cls in enumerate(sorted(model["clsDic"]))}
    # Model 1
    # Here we sum the probabilities of the words inside the tercet for each class probability
    # S is a Matrix of shape (NumClasses, NumSample(NumTercets))
    S = numpy.zeros((len(hCls2Idx), len(lTercets)))
    for tIdx, tercet in enumerate(lTercets):
        # hScores is an array (1, NulClasses) in which for each class there is the sum of probabilities
        hScores = S1_compute_logLikelihoodsMy(model, tercet)
        # We sort the class labels so that rows are ordered according to alphabetical order of labels
        for cls in hCls2Idx:
            S[hCls2Idx[cls], tIdx] = hScores[model["clsDic"][cls]]

    # Model 2 quick solution, with matrix product
    # Here we count the occurrences of the words inside the tercet, then we do the scalar product
    # (NumClasses, NumDistinctWords) x (NumOccurencesTercet, NumTercet) = (NumClasses, NumTercet) that is the S matrix
    # Sort the model rows
    modelSort = numpy.zeros((len(hCls2Idx), len(model["globalDic"])))
    for cls in hCls2Idx:
        modelSort[hCls2Idx[cls], :] = model["clsLogProb"][model["clsDic"][cls]]
    # occTerc matrix of shape (NumDistWords, NumSample(NumTercets))
    # in each cell there is the occurrence of a word in the tercet
    occTerc = numpy.zeros((len(model["globalDic"]), len(lTercets)))
    for tIdx, tercet in enumerate(lTercets):
        for word in tercet.split():
            if word in model["globalDic"]:
                occTerc[model["globalDic"][word], tIdx] += 1

    S = numpy.dot(modelSort, occTerc)

    return S


def compute_classPosteriors(S, logPrior=None):
    """
    Compute class posterior probabilities

    S: Matrix of class-conditional log-likelihoods
    logPrior: array with class prior probability (shape (#cls, ) or (#cls, 1)). If None, uniform priors will be used

    Returns: matrix of class posterior probabilities
    """

    if logPrior is None:
        logPrior = numpy.log(numpy.ones(S.shape[0]) / float(S.shape[0]))

    J = S + mcol(logPrior)  # Compute joint probability
    ll = scipy.special.logsumexp(J, axis=0)  # Compute marginal likelihood log f(x)
    P = J - ll  # Compute posterior log-probabilities P = log ( f(x, c) / f(x)) = log f(x, c) - log f(x)
    return numpy.exp(P)


def compute_accuracy(P, L):
    """
    Compute accuracy for posterior probabilities P and labels L. L is the integer associated to the correct label (in alphabetical order)
    """

    PredictedLabel = numpy.argmax(P, axis=0)
    NCorrect = (PredictedLabel.ravel() == L.ravel()).sum()
    NTotal = L.size
    return float(NCorrect) / float(NTotal)


if __name__ == "__main__":
    # they are arrays of line, thus lines
    lInf, lPur, lPar = load.load_data()

    lInf_train, lInf_eval = load.split_data(lInf, 4)
    lPur_train, lPur_eval = load.split_data(lPur, 4)
    lPar_train, lPar_eval = load.split_data(lPar, 4)

    # Solution 1
    # Multiclass

    hCls2Idx = {'inferno': 0, 'purgatorio': 1, 'paradiso': 2}

    hlTrain = {
        'inferno': lInf_train,
        'purgatorio': lPur_train,
        'paradiso': lPar_train
    }

    l_eval = lInf_eval + lPur_eval + lPar_eval

    S1_model = S1_estimateModelMy(hlTrain, eps=0.001)

    S1_P = compute_classPosteriors(S1_compute_logLikelihoodMatrixMy(
        S1_model,
        l_eval,
        hCls2Idx,
    ),
        numpy.log(numpy.array([1. / 3, 1. / 3, 1. / 3]))
    )

    labelsInf = numpy.zeros(len(lInf_eval))
    labelsInf[:] = hCls2Idx['inferno']

    labelsPar = numpy.zeros(len(lPar_eval))
    labelsPar[:] = hCls2Idx['paradiso']

    labelsPur = numpy.zeros(len(lPur_eval))
    labelsPur[:] = hCls2Idx['purgatorio']

    labelsEval = numpy.hstack([labelsInf, labelsPur, labelsPar])

    print(compute_accuracy(S1_P[:, labelsEval == hCls2Idx['inferno']], labelsEval[labelsEval == hCls2Idx['inferno']]))
    print(compute_accuracy(S1_P[:, labelsEval == hCls2Idx['purgatorio']], labelsEval[labelsEval == hCls2Idx['purgatorio']]))
    print(compute_accuracy(S1_P[:, labelsEval == hCls2Idx['paradiso']], labelsEval[labelsEval == hCls2Idx['paradiso']]))
    print(compute_accuracy(S1_P, labelsEval))
