import numpy
import scipy.special
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

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

def logpdf_GAU_ND(X, mu, C):
    M = mu.shape[0]  # dimensions
    pi = numpy.pi
    Precision = numpy.linalg.inv(C)

    first = -M / 2 * numpy.log(2 * pi)
    second = -0.5 * numpy.linalg.slogdet(C)[1]
    XC = X - mu  # center the values
    third_1 = numpy.dot(XC.T, Precision)
    third = -0.5 * (third_1.T * XC).sum(0)
    return first + second + third

def MVG_Classifier_Model(DTR, LTR, K):
    D = DTR.shape[0]
    u = numpy.zeros((D, K))  # array of means vectors by class [MATRIX (D, K)]
    C = numpy.zeros((K, D, D))  # array of covariance matrices by class
    ## MULTIVARIATE GAUSSIAN CLASSIFIER
    for i in numpy.arange(K):
        ## ESTIMATION OF MODEL
        # only the class i
        DTRi = DTR[:, LTR == i]  # Matrix of samples of class i (D, Ni)
        # compute the mean
        u[:, i:i+1] = mcol(DTRi.mean(1))  # vector of means of dimensions of class i (D, 1)
        # center the points
        DTRiC = DTRi - u[:, i:i+1]
        # compute the covariance matrix
        C[i] = numpy.dot(DTRiC, DTRiC.T)/DTRiC.shape[1]
    return u, C

def MVG_NaiveBayes_Classifier_Model(DTR, LTR, K):
    u = numpy.zeros((D, K))  # array of means vectors by class [MATRIX (D, K)]
    C = numpy.zeros((K, D, D))  # array of covariance matrices by class
    for i in range(K):
        # take the samples of class i
        DTRi = DTR[:, LTR == i]
        # compute the mean
        u[:, i:i+1] = mcol(DTRi.mean(1))  # vector of means of dimensions of class i (D, 1)
        # center the points
        DTRiC = DTRi - u[:, i:i+1]
        # compute the covariance matrix fast method
        C[i] = numpy.diag((DTRiC**2).sum(1))/DTRiC.shape[1]
    return u, C

def Tied_MVG_Classifier_Model(DTR, LTR, K):
    D = DTR.shape[0]  # Dimensions of the dataset
    u = numpy.zeros((D, K))  # array of means vectors by class [MATRIX (D, K)]
    C = numpy.zeros((D, D))  # covariance matrix initialization
    for i in range(K):
        # take the samples of class i
        DTRi = DTR[:, LTR == i]
        # compute the mean
        u[:, i:i+1] = mcol(DTRi.mean(1))  # vector of means of dimensions of class i (D, 1)
        # center the points
        DTRiC = DTRi - u[:, i:i+1]
        # compute the partial covariance matrix and add it to within-class
        C += numpy.dot(DTRiC, DTRiC.T)
    # divide the partial covariance by the number of samples
    C /= DTR.shape[1]
    return u, C

def Tied_NaiveBayes_Classifier_Model(DTR, LTR, K):
    D = DTR.shape[0]  # Dimensions of the dataset
    u = numpy.zeros((D, K))  # array of means vectors by class [MATRIX (D, K)]
    C = numpy.zeros((D, D))  # covariance matrix
    for i in range(K):
        # take the samples of class i
        DTRi = DTR[:, LTR == i]
        # compute the mean
        u[:, i:i+1] = mcol(DTRi.mean(1))  # vector of means of dimensions of class i (D, 1)
        # center the points
        DTRiC = DTRi - u[:, i:i+1]
        # compute the partial covariance matrix fast method
        C += numpy.diag((DTRiC**2).sum(1))
    # divide the partial covariance by the number of samples
    C /= DTR.shape[1]
    return u, C

def inference(DTE, K, u, C, fullCov, logPrior=None):
    # Proportional probabilities if they are not provided
    if logPrior is None:
        logPrior = numpy.log(numpy.ones(K) / float(K))

    N = DTE.shape[1]  # N test set
    logSJoint = numpy.zeros((K, N))  # (K, Ntest) array of joint probability by class then vetical Stacked
    for i in numpy.arange(K):
        # compute the likelihoods function
        if fullCov:
            S = mrow(logpdf_GAU_ND(DTE, u[:, i:i+1], C[i]))
        else:
            S = mrow(logpdf_GAU_ND(DTE, u[:, i:i+1], C))
        # compute the join distribution
        logSJoint[i] = S+logPrior[i]

    # caclulate the log-sum-exp trick
    logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
    # compute the posterior prob
    logSPost = logSJoint - logSMarginal
    SPostLog = numpy.exp(logSPost)
    return SPostLog


def compute_confusion_matrix_oN(P, L):
    # P => Predictions
    # L => Labels
    K = L.max()+1
    M = numpy.zeros((K, K))
    for i in numpy.arange(L.shape[0]):
        M[P[i], L[i]] += 1
    return M

def compute_confusion_matrix_oK(P, L):
    # P => Predictions
    # L => Labels
    K = L.max()+1
    M = numpy.zeros((K, K))
    for i in numpy.arange(K):
        for j in numpy.arange(K):
            M[i, j] = ((P == i) * (L == j)).sum()
    return M

def posterior_prob(ll, pi):
    logPrior = numpy.log(pi)
    logJoint = ll + logPrior
    logMarginal = scipy.special.logsumexp(logJoint, axis=0)
    return numpy.exp(logJoint - logMarginal)

def assign_label_multiclass_opt_cost(ll, C, pi):
    P = posterior_prob(ll, pi)
    C_prob = numpy.dot(C, P)
    # matrix of Cost (K, K) x matrix of log likelihood (K, N)
    # dot product row by colum to retrieve the cost of chose that class prediction

    # we will choose the prediction with the minimum cost
    return numpy.argmin(C_prob, axis=0)

def assign_label_bin(llr, p, Cfn, Cfp, t=None):
    if t is None:
        t = - numpy.log(p * Cfn / ((1 - p) * Cfp))
    P = (llr > t) + 0  # + 0 converts booleans to integers
    return P

def DCF_u_mult(ll, L, p, C):
    P = assign_label_multiclass_opt_cost(ll, C, p)
    M = compute_confusion_matrix_oK(P, L)
    # colum by colum dot product between C and M, for doing this I need to transpose the matrix C
    # and take only the diagonal: I want only col1xcol1 col2xcol2 col3xcol3 ...
    optC = numpy.diag(numpy.dot(C.T, M))
    # compute the number of sample for each class
    N = numpy.sum(M, axis=0)
    # apply the formula, and sum all
    return (optC*p.T/N).sum()

def DCF_u_bin(llr, L, p, Cfn, Cfp, t=None):
    P = assign_label_bin(llr, p, Cfn, Cfp, t)
    M = compute_confusion_matrix_oK(P, L)

    TN, FN, FP, TP = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    FNR = FN/(FN + TP)
    FPR = FP/(FP + TN)
    B = p*Cfn*FNR + (1-p)*Cfp*FPR
    return B

def DCF_norm_mult(ll, L, p, C):
    B = DCF_u_mult(ll, L, p, C)
    BDummy = numpy.min(numpy.dot(C, p))
    return B/BDummy

def DCF_norm_bin(llr, L, p, Cfn, Cfp, t=None):
    B = DCF_u_bin(llr, L, p, Cfn, Cfp, t)
    BDummy = min(p*Cfn, (1 - p)*Cfp)
    return B/BDummy

def DCF_min(llr, L, p, Cfn, Cfp):
    # llr => log-likelihood ratio ; p => prior prob; Cfn => FNR; Cfp => FPR
    ts = numpy.array(llr)
    ts.sort()
    ts = numpy.concatenate((numpy.array([-numpy.inf]), ts, numpy.array([+numpy.inf])))

    B_norm = []
    for t in ts:
        B_norm.append( DCF_norm_bin(llr, L, p, Cfn, Cfp, t) )
    return numpy.array(B_norm).min()

def print_ROC(llr, L):
    ts = numpy.array(llr)
    ts.sort()
    ts = numpy.concatenate((numpy.array([-numpy.inf]), ts, numpy.array([+numpy.inf])))
    x = []
    y = []
    for t in ts:
        P = (llr > t) + 0
        M = compute_confusion_matrix_oK(P, L)
        TN, FN, FP, TP = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        FPR = FP/(TN+FP)
        FNR = FN/(FN+TP)
        TPR = 1 - FNR
        x.append(FPR)
        y.append(TPR)
    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.scatter(x, y, s=4)
    plt.show()

def bayes_error_addToPlot(llr, L, title):
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    mindcf = []
    dcf = []
    for effPriorLogOdd in effPriorLogOdds:
        effPrior = 1/(1+numpy.exp(-effPriorLogOdd))
        dcf.append(DCF_norm_bin(llr, L, effPrior, 1, 1))  # actually
        mindcf.append(DCF_min(llr, L, effPrior, 1, 1))  # minimum

    plt.plot(effPriorLogOdds, dcf, label="DCF "+title)
    plt.plot(effPriorLogOdds, mindcf, label="min DCF "+title)

if __name__ == "__main__":
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    D = DTR.shape[0]  # Dimensions
    K = LTR.max()+1  # Classes

    ## ------------------------------

    U, C = MVG_Classifier_Model(DTR, LTR, K)
    S = inference(DTE, K, U, C, True)
    PredictedLabel = numpy.argmax(S, axis=0)

    # Confusion matrix MVG model
    M_MVG = numpy.zeros((K, K))
    for i in numpy.arange(LTE.shape[0]):
        M_MVG[PredictedLabel[i], LTE[i]] += 1
    print(M_MVG)

    ## ------------------------------

    U, C = MVG_NaiveBayes_Classifier_Model(DTR, LTR, K)
    S = inference(DTE, K, U, C, True)
    PredictedLabel = numpy.argmax(S, axis=0)

    # Confusion matrix MVG model
    M_MVG_Naive = numpy.zeros((K, K))
    for i in numpy.arange(LTE.shape[0]):
        M_MVG_Naive[PredictedLabel[i], LTE[i]] += 1
    print(M_MVG_Naive)

    ## ------------------------------

    U, C = Tied_MVG_Classifier_Model(DTR, LTR, K)
    S = inference(DTE, K, U, C, False)
    PredictedLabel = numpy.argmax(S, axis=0)

    # Confusion matrix Tied MVG model
    M_Tied = numpy.zeros((K, K))
    for i in numpy.arange(LTE.shape[0]):
        M_Tied[PredictedLabel[i], LTE[i]] += 1
    print(M_Tied)

    ## ------------------------------

    U, C = Tied_NaiveBayes_Classifier_Model(DTR, LTR, K)
    S = inference(DTE, K, U, C, False)
    PredictedLabel = numpy.argmax(S, axis=0)

    # Confusion matrix Tied MVG model
    M_Tied_Naive = numpy.zeros((K, K))
    for i in numpy.arange(LTE.shape[0]):
        M_Tied_Naive[PredictedLabel[i], LTE[i]] += 1
    print(M_Tied_Naive)

    ## -----------------------------

    LTE = numpy.load("./Data/commedia_labels.npy")
    ll = numpy.load("./Data/commedia_ll.npy")  # (K, N)
    logPrior = numpy.log(numpy.ones(K) / float(K))

    J = ll - mcol(logPrior)
    marginal = scipy.special.logsumexp(J, axis=0)
    S = J - marginal
    PredictedLabel = numpy.argmax(S, axis=0)

    # Confusion matrix Tied MVG model
    M_commedia = numpy.zeros((K, K))
    for i in numpy.arange(LTE.shape[0]):
        M_commedia[PredictedLabel[i], LTE[i]] += 1
    print(M_commedia)

    ## ---------------------------------------

    llr_inf_par = numpy.load("./Data/commedia_llr_infpar.npy")
    labels_inf_par = numpy.load("./Data/commedia_labels_infpar.npy")
    llr_inf_par_eps1 = numpy.load("Data/commedia_llr_infpar_eps1.npy")
    labels_inf_par_eps1 = numpy.load("Data/commedia_labels_infpar_eps1.npy")

    print("(0.5, 1, 1)")
    P = assign_label_bin(llr_inf_par, 0.5, 1, 1)
    M = compute_confusion_matrix_oK(P, labels_inf_par)
    print(M)
    B = DCF_u_bin(llr_inf_par, labels_inf_par, 0.5, 1, 1)
    B = DCF_norm_bin(llr_inf_par, labels_inf_par, 0.5, 1, 1)
    print(B)
    B_min = DCF_min(llr_inf_par, labels_inf_par, 0.5, 1, 1)
    print(B_min)
    B = DCF_norm_bin(llr_inf_par_eps1, labels_inf_par_eps1, 0.5, 1, 1)
    print(B)
    B_min = DCF_min(llr_inf_par_eps1, labels_inf_par_eps1, 0.5, 1, 1)
    print(B_min)

    print("(0.8, 1, 1)")
    P = assign_label_bin(llr_inf_par, 0.8, 1, 1)
    M = compute_confusion_matrix_oK(P, labels_inf_par)
    print(M)
    B = DCF_u_bin(llr_inf_par, labels_inf_par, 0.8, 1, 1)
    B = DCF_norm_bin(llr_inf_par, labels_inf_par, 0.8, 1, 1)
    print(B)
    B_min = DCF_min(llr_inf_par, labels_inf_par, 0.8, 1, 1)
    print(B_min)
    B = DCF_norm_bin(llr_inf_par_eps1, labels_inf_par_eps1, 0.8, 1, 1)
    print(B)
    B_min = DCF_min(llr_inf_par_eps1, labels_inf_par_eps1, 0.8, 1, 1)
    print(B_min)

    print("(0.5, 10, 1)")
    P = assign_label_bin(llr_inf_par, 0.5, 10, 1)
    M = compute_confusion_matrix_oK(P, labels_inf_par)
    print(M)
    B = DCF_u_bin(llr_inf_par, labels_inf_par, 0.5, 10, 1)
    B = DCF_norm_bin(llr_inf_par, labels_inf_par, 0.5, 10, 1)
    print(B)
    B_min = DCF_min(llr_inf_par, labels_inf_par, 0.5, 10, 1)
    print(B_min)
    B = DCF_norm_bin(llr_inf_par_eps1, labels_inf_par_eps1, 0.5, 10, 1)
    print(B)
    B_min = DCF_min(llr_inf_par_eps1, labels_inf_par_eps1, 0.5, 10, 1)
    print(B_min)

    print("(0.8, 1, 10)")
    P = assign_label_bin(llr_inf_par, 0.8, 1, 10)
    M = compute_confusion_matrix_oK(P, labels_inf_par)
    print(M)
    B = DCF_u_bin(llr_inf_par, labels_inf_par, 0.8, 1, 10)
    B = DCF_norm_bin(llr_inf_par, labels_inf_par, 0.8, 1, 10)
    print(B)
    B_min = DCF_min(llr_inf_par, labels_inf_par, 0.8, 1, 10)
    print(B_min)
    B = DCF_norm_bin(llr_inf_par_eps1, labels_inf_par_eps1, 0.8, 1, 10)
    print(B)
    B_min = DCF_min(llr_inf_par_eps1, labels_inf_par_eps1, 0.8, 1, 10)
    print(B_min)

    '''
    + When the prior for class 1 increases, the classifier tends to predict class 1 more frequently
    + When the cost of predicting class 0 when the actual class is 1, C0;1 increases, the classifier will
    make more false positive errors and less false negative errors. The opposite is true when C1;0 is
    higher.
    '''

    '''
    The Bayes risk allows us comparing different systems, however it does not tell us what is the benefit of
    using our recognizer with respect to optimal decisions based on prior information only. We can compute
    a normalized detection cost, by dividing the Bayes risk by the risk of an optimal system that does not
    use the test data at all. We have seen that the cost of such system is
    '''

    print_ROC(llr_inf_par, labels_inf_par)

    plt.figure()
    plt.xlabel("prior lod-odds")
    plt.ylabel("DCF value")
    bayes_error_addToPlot(llr_inf_par, labels_inf_par, "eps 0.0001")
    bayes_error_addToPlot(llr_inf_par_eps1, labels_inf_par_eps1, "eps 1")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.legend()
    plt.show()

    ## MULTICLASS PROBLEM ##
    ll_commedia = numpy.load("Data/commedia_ll.npy")
    ll_commedia_eps1 = numpy.load("Data/commedia_ll_eps1.npy")
    labels_commedia = numpy.load("Data/commedia_labels.npy")
    labels_commedia_eps1 = numpy.load("Data/commedia_labels_eps1.npy")
    C = numpy.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    pi = mcol(numpy.array([0.3, 0.4, 0.3]))

    P = assign_label_multiclass_opt_cost(ll_commedia, C, pi)
    M = compute_confusion_matrix_oK(P, labels_commedia)
    print(M)
    B = DCF_u_mult(ll_commedia, labels_commedia, pi, C)
    print(B)
    B = DCF_norm_mult(ll_commedia, labels_commedia, pi, C)
    print(B)

    P = assign_label_multiclass_opt_cost(ll_commedia_eps1, C, pi)
    M = compute_confusion_matrix_oK(P, labels_commedia_eps1)
    print(M)
    B = DCF_u_mult(ll_commedia_eps1, labels_commedia_eps1, pi, C)
    print(B)
    B = DCF_norm_mult(ll_commedia_eps1, labels_commedia_eps1, pi, C)
    print(B)

    # normal prio and cost
    print()
    C = numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pi = mcol(numpy.array([1.0/3, 1.0/3, 1.0/3]))

    B = DCF_u_mult(ll_commedia, labels_commedia, pi, C)
    print(B)
    B = DCF_norm_mult(ll_commedia, labels_commedia, pi, C)
    print(B)

    B = DCF_u_mult(ll_commedia_eps1, labels_commedia_eps1, pi, C)
    print(B)
    B = DCF_norm_mult(ll_commedia_eps1, labels_commedia_eps1, pi, C)
    print(B)












