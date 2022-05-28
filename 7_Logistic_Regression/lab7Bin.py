import scipy.optimize
import numpy

def mcol(x):
    return x.reshape((x.size, 1))

def f(x):
    y, z = x[0], x[1]
    return (y + 3) ** 2 + numpy.sin(y) + (z + 1) ** 2

def fGrad(x):
    # The optimizer will do less iteration, and will be more accurate
    y, z = x[0], x[1]
    obj = (y + 3) ** 2 + numpy.sin(y) + (z + 1) ** 2
    # (df/dy, df/dz) GRADIENT
    grad = numpy.array([2 * (y + 3) + numpy.cos(y), 2 * (z + 1)])
    return obj, grad

def load_iris_binary():
    import sklearn.datasets
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    # set the seed
    numpy.random.seed(seed)
    # create a vector (,1) of random number no repetitions
    idx = numpy.random.permutation(D.shape[1])
    # divide the random numbers in 2 parts
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    # get only the samples of that random number
    DTR = D[:, idxTrain]
    LTR = L[idxTrain]
    DTE = D[:, idxTest]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)

def logreg_obj_wrap(DTR, LTR, lamb):
    Z = (LTR * 2.0) - 1.0  # ci = 1 => 1 ; ci = 0 => -1
    M = DTR.shape[0]
    def logreg_obj(v):
        # J(w, b), in v there are D+1 elements, D element of array w, and the cost b
        """
        Compute and return the objective function value using DTR,
        LTR, l
        """
        w, b = v[0:M], v[-1]
        w = mcol(w)  # (D, 1) D := Dimensions
        """
        Here we are computing 1/n * sum (log (1 + exp(-z1*(w.T * xi + b)) ))
        The computation of logarithm can lead to numerical issues, so we use numpy.logaddexp
        the function compute log(exp(a) + exp(b)) 
            1 => exp(0)
        """
        S = numpy.dot(w.T, DTR) + b  # The second operand of  exp(...)
        obj = numpy.logaddexp(0, -Z*S).mean()  # Computing the log AND doing 1/n * sum(..), practically the mean
        # The norm^2 can be calculated also with:  (w**2).sum()
        return lamb/2 * numpy.linalg.norm(w)**2 + obj
    return logreg_obj

def compute_accuracy(P, L):
    """
    Compute accuracy for posterior probabilities P and labels L. L is the integer associated to the correct label
    (in alphabetical order)
    """

    NCorrect = (P.ravel() == L.ravel()).sum()
    NTotal = L.size
    return float(NCorrect) / float(NTotal)


if __name__ == "__main__":
    """
    The optimizer will calculate in own method the gradient, it will take more iterations and the 
    precision is less accurate
    """
    x, f, d = scipy.optimize.fmin_l_bfgs_b(f, numpy.zeros(2), approx_grad=True, iprint=1)
    print(x)  # x is the estimated position of the minimum
    print(f)  # f is the objective value at the minimum
    print(d)  # d contains additional information

    """
    The optimizer will do less iteration, and will be more accurate
    Here the optimizer use the gradient that you have calculated, so it's more precise
    """
    x, f, d = scipy.optimize.fmin_l_bfgs_b(fGrad, numpy.zeros(2), iprint=1)
    print(x)
    print(f)
    print(d)

    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    for lamb in [10**-6]:
        x, f, d = scipy.optimize.fmin_l_bfgs_b(
                            logreg_obj_wrap(DTR, LTR, lamb),
                            numpy.zeros(DTR.shape[0] + 1),
                            approx_grad=True
                    )
        print(x)
        print(f)
        wBest = mcol(x[0: DTR.shape[0]])  # (D, 1)
        bBest = x[-1]  # scalar
        """
        Here we compute the score vector, is a vector with NumSample elements
        it's computed by :      s(xt) = w.T * xt + b
        """
        S = numpy.dot(wBest.T, DTE) + bBest  # (1, NumSamples)
        """
        s[i] > 0 ==> LP[i] = 1 ; else LP[i] = 0
        """
        LP = S > 0
        acc = compute_accuracy(LP, LTE)  # I take this method from the lab5 with a little change
        print(1-acc)
