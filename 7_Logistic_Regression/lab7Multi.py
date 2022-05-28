import scipy.optimize
import scipy.special
import numpy

def mcol(x):
    return x.reshape((x.size, 1))

def mrow(x):
    return x.reshape((1, x.size))

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

def load_iris():
    import sklearn.datasets
    return sklearn.datasets.load_iris()["data"].T, sklearn.datasets.load_iris()["target"]

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
    """
    Use the 1-of-K encoding of the labels:
    a = array([1,0,3]) =>
    b = array([[0,1,0,0], [1,0,0,0], [0,0,0,1]])
    """
    Z = numpy.zeros((LTR.size, LTR.max()+1))
    Z[numpy.arange(LTR.size), LTR] = 1
    Z = Z.T  # (K, N)

    D = DTR.shape[0]  # num Dimensions
    K = LTR.max()+1  # num Classes
    def logreg_obj(v):
        # J(w, b), in v there are D+1 elements, D element of array w, and the cost b
        """
        Compute and return the objective function value using DTR,
        LTR, l
        """
        # Conversion of the inputs
        W, b = v[0:D*K], v[D*K:]
        W = W.reshape((D, K))  # (D, K) D := Dimensions;  K := Classes
        b = b.reshape((K, 1))  # (K, 1)
        """
        Compute the matrix of scores S: S_ki should be equal to S_ki = w_k^T * x_i + b_k,
        """
        S = numpy.dot(W.T, DTR) + b  # Score matrix (K, N)
        """
        Compute matrix Y log containing Y_ki^log = log( y_ik )
        Same process as when you compute the class posterior prob in Generative Models
        """
        logsumexp = mrow(scipy.special.logsumexp(S, axis=0))  # (1, N) Compute the log(sum(exp( )))
        Ylog = S - logsumexp  # (K, N), it's S but for each row is subtracted the logsumexp
        obj = ((Z*Ylog).sum(axis=0)).mean()

        # The norm^2 can be calculated also with:  (w**2).sum() or (W*W).sum()
        return lamb/2 * (W*W).sum() - obj
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

    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    D = DTR.shape[0]  # Number of dimensions
    K = LTE.max()+1  # Number of classes

    for lamb in [10**-6, 10**-3, 10**-1, 1.0]:
        x, f, d = scipy.optimize.fmin_l_bfgs_b(
                            logreg_obj_wrap(DTR, LTR, lamb),
                            numpy.zeros(DTR.shape[0] * K + K),
                            # W has shape (D, K), but the function accept only 1-D numpy array
                            # same for b, it has shape (K, 1)
                            approx_grad=True
                    )
        print(x)
        print(f)
        # Conversion of the inputs
        WBest, bBest = x[0:D * K], x[D * K:]
        WBest = WBest.reshape((D, K))  # (D, K) D := Dimensions;  K := Classes
        bBest = bBest.reshape((K, 1))  # (K, 1)
        """
        Here we compute the score matrix
        it's computed by :      S = W.T * X + B
        """
        S = numpy.dot(WBest.T, DTE) + bBest  # (K , NumSamples)
        LP = S.argmax(0)  # Choose the class with the maximum Score

        acc = compute_accuracy(LP, LTE)  # I take this method from the lab5 with a little change
        print(1 - acc)
