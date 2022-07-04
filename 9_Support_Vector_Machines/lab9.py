import numpy
import scipy.optimize

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

def m1D(v):
    return v.reshape(v.size)


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


def addKfactor(D, K):
    """
    Add the K factor as last row of the Data
    it's like an additional dimension that is equal for all samples

    at the end x_i = (x_i K).T , for each sample
    """
    return numpy.vstack((D, numpy.full((1, D.shape[1]), K)))

def noKern(D1, D2, **hyperPars):
    """
    without kernel, it's only the dot product
    """
    return numpy.dot(D1.T, D2)  # (N, N)

def polyKern(D1, D2, **hyperPars):
    c = hyperPars["c"]
    d = hyperPars["d"]
    eps = hyperPars["K"] ** 2
    """
    polynomial kernel of degree d
    k(x_1, x_2) = (x_1.T * x_2 + c)^d
    """
    return (numpy.dot(D1.T, D2) + c)**d + eps  # (N, N)

def radialBasisKern(D1, D2, **hyperPars):
    g = hyperPars["g"]
    eps = hyperPars["K"]**2
    """
    Radial basis function kernel
    K(x_1, x_2) = exp(-g*||x_1-x_2||^2)
    """
    return numpy.exp(-g*numpy.linalg.norm(D1-D2)**2) + eps

def computeH(Z, D1, D2, kern, **hyperPars):
    """
    compute the matrix H as in the formula:
    H = z_i * z_j * x_i.T * x_j
    H is (N, N)
    """
    if hyperPars and hyperPars["d"] is not None and hyperPars["c"] is not None:
        G = kern(D1, D2, d=hyperPars["d"], c=hyperPars["c"], g=hyperPars["g"], K=K)
    elif hyperPars and hyperPars["g"] is not None:
        G = kern(D1, D2, g=hyperPars["g"], K=K)
    else:
        G = kern(D1, D2)
    # compute the kernel (N, N)
    Z = numpy.dot(Z.T, Z)  # (N, N) each cell is z_i * z_j
    return Z*G  # (N, N)

def convertToPrimal(D, Z, aBest):
    """
    convert to primal solution
    w = summatory( a_i * z_i * x_i )
    """
    return numpy.dot(D, Z.T * aBest)

def SVGmodel(DTR, LTR, C, K, kern, **hyperPars):
    # starting point to search the minimum: Origin
    initialValues = numpy.zeros((DTR.shape[1], 1))
    # list of tuples [(0, C), (0, C), ..., (0, C)]
    constraints = [(0, C)] * DTR.shape[1]

    D = addKfactor(DTR, K)  # +1 row with all K
    Z = mrow((LTR * 2.0) - 1.0)  # 0 => -1 ; 1 => 1
    if hyperPars and hyperPars["d"] is not None and hyperPars["c"] is not None:
        H = computeH(Z, D, D, kern, d=hyperPars["d"], c=hyperPars["c"], K=K)
    elif hyperPars and hyperPars["g"] is not None:
        H = computeH(Z, D, D, kern, g=hyperPars["g"], K=K)
    else:
        H = computeH(Z, D, D, kern)
    # (N, N) z_i * z_j * x_i.T * x_j

    def JDual(a):
        a = mcol(a)
        """
        Compute the formula for the lagrangian
        J^D(a) = - 1/2*a.T*H*a + a.T*1
        return a lagrangian and gradient
        """
        Ha = numpy.dot(H, a)
        aHa = numpy.dot(a.T, Ha)
        a1 = a.sum()
        lagrangian = -0.5 * aHa + a1

        '''
        Compute the gradient of Lagrangian
        Gradient = - H*a + 1        
        '''
        gradient = -Ha + 1

        return lagrangian, gradient

    def LDual(a):
        """
        Compute the formula for the lagrangian
        L^D(a) = -J^D(a)
        return a lagrangian and the gradient
        """
        loss, grad = JDual(a)
        return -loss, -grad

    def JPrimal(w):
        S = numpy.dot(w.T, D)
        loss = numpy.maximum(0, 1-Z*S).sum()
        return 0.5 * numpy.linalg.norm(w)**2 + C * loss

    '''
    Find the maximum of Dual solution J(a) => finding the minimum of L(a) = -J(a)
    '''
    x, f, d = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        initialValues,
        bounds=constraints,
        factr=0.0,
        maxiter=100000,  # maximum number of allowed iterations
        maxfun=100000  # maximum number of calls to the obj function
    )
    aBest = mcol(x)  # (N, 1)
    """
    use the dual solution in order to find the primal solution
    w = summatory( a_i * z_i * x_i )
    """
    wBest = convertToPrimal(D, Z, aBest)

    primalLoss = JPrimal(wBest)
    dualLoss = JDual(aBest)[0].sum()
    gap = JPrimal(wBest) - dualLoss
    print(f'{K:3}\t{C:3}\t{primalLoss:20}\t{dualLoss:20}\t{gap:20}\t', end="")

    return wBest, aBest


if __name__ == "__main__" :
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    print(f'{"K":3}\t{"C":3}\t{"Primal loss":20}\t{"Dual loss":20}\t{"Duality gap":20}\t{"Error rate":20}')
    for C, K in [(0.1, 1), (1, 1), (10, 1), (0.1, 10), (1, 10), (10, 10)]:
        wb, a = SVGmodel(DTR, LTR, C, K, noKern)

        """
        Here we compute the score vector, is a vector with NumSample elements
        it's computed by :      s(xt) = w.T * xt + b
        """
        # wb = (D+1, 1) extract the b term
        w = mcol(wb[0:DTR.shape[0]])  # (D, 1)
        b = wb[-1]  # (1, 1)

        Sw = numpy.dot(w.T, DTE) + b
        Z = mrow((LTR * 2.0) - 1.0)
        Sa = mrow(numpy.dot(noKern(DTE, DTR), Z.T * a))
        P = Sw > 0  # Th threshold should be chosen through a Bayes decision
        Pa = Sa > 0

        NCorrect = (P.ravel() == LTE.ravel()).sum()
        NTotal = LTE.size
        accuracy = float(NCorrect) / float(NTotal)

        print(f'{(1-accuracy)*100:20} {(Sw-Sa).sum()}\n', end="")
