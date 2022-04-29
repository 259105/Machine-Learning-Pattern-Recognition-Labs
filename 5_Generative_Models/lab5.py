from asyncio.windows_events import NULL
import numpy;
import scipy;
import scipy.special;

def mrow(v) :
    return v.reshape((1,v.size));

def mcol(v) :
    return v.reshape((v.size,1));


def loadIrisDataset() :
    import sklearn.datasets;
    return sklearn.datasets.load_iris()["data"].T, sklearn.datasets.load_iris()["target"];

def load(file):
    datasetFlowers = [];
    labels = [];
    dicFlowers = {
        "Iris-setosa" : 0,
        "Iris-versicolor" : 1,
        "Iris-virginica" : 2,
    }
    with open(file) as f :
        for line in f :
            try : ## could be wrong lines, with this we avoid this problems
                fields = line.split(",");
                attrs = [float(s) for s in fields[0:4]];
                datasetFlowers.append(mcol(numpy.array(attrs)));
                labels.append(dicFlowers[fields[-1].strip()]);
            except :
                pass;
        return numpy.hstack(datasetFlowers), numpy.array(labels, dtype=numpy.int32);    

def split_db_2to1(D, L, seed=0) : 
    nTrain = int(D.shape[1]*2.0/3.0);
    # set the seed
    numpy.random.seed(seed);
    # create a vector (,1) of random number no repetitions
    idx = numpy.random.permutation(D.shape[1]);
    # divide the random numbers in 2 parts
    idxTrain = idx[0:nTrain];
    idxTest = idx[nTrain:];
    # get only the samples of that random number
    DTR = D[:,idxTrain];
    LTR = L[idxTrain];
    DTE = D[:,idxTest];
    LTE = L[idxTest];

    return (DTR,LTR), (DTE,LTE);

def logpdf_GAU_ND(x, mu, C) :
    first = -mu.shape[0]/2*numpy.log(2*numpy.pi);
    second = -1/2*numpy.linalg.slogdet(C)[1];
    xc = x - mu; # center the values
    third = -1/2*numpy.dot(numpy.dot(xc.T,numpy.linalg.inv(C)),xc);
    # take only the rows (i,i) 
    return numpy.diagonal(first+second+third);

def MVG_Classifier(DTR, LTR, DTE, LTE) :
    print("MVG-Classifier")
    u = []; # array of means vectors by class
    C = []; # array of covariance matrices by class
    PrioP = [1/3, 1/3, 1/3] # Prior prob of classes
    SJoint = [] ;
    ## MULTIVARIATE GAUSSIAN CLASSIFIER
    for i in [0,1,2] : 
        ## ESTIMATION OF MODEL
        # only the class i
        DTRi = DTR[:, LTR==i];
        # compute the mean
        u.append(mcol(DTRi.mean(1)));
        # center the points
        DTRiC = DTRi - u[i];
        # compute the covariance matrix
        C.append(numpy.dot(DTRiC,DTRiC.T)/DTRiC.shape[1]);
        # print(u[i]); print(C[i]);

        ## INFERENCE FOR TEST SAMPLE
        # compute the likelihoods function
        S = mrow(numpy.exp(logpdf_GAU_ND(DTE,u[i],C[i])));
        # compute the join distribution
        SJoint.append(S*PrioP[i]);
    # vertically stack the SJoint array into numpy.ndarray
    SJoint = numpy.vstack(SJoint);
    # compute the marginal denisty
    SMarginal = mrow(SJoint.sum(0));
    # compute the posterior prob
    SPost = SJoint/SMarginal;
    # find the predicted label
    predClass = SPost.argmax(0);
    #print(predClass); print(LTE);

    ## ACCURACY OF MODEL
    # (num Wrong prediction)/(num of samples)
    acc = (predClass!=LTE).sum() / LTE.size;
    print("Accuracy :")
    print(acc);
    print("Checks:")
    print((numpy.load("Solution/SJoint_MVG.npy")-SJoint).sum());
    print((numpy.load("Solution/Posterior_MVG.npy")-SPost).sum());

    ## AGAIN BUT WITH LOG
    print("Log-densities:")
    logSJoint = [];
    for i in range(3) :
        S = mrow(logpdf_GAU_ND(DTE,u[i],C[i]));
        logSJoint.append(S+numpy.log(PrioP[i]));
    logSJoint = numpy.vstack(logSJoint);
    # caclulate the log-sum-exp trick
    logSMarginal = mrow(scipy.special.logsumexp(logSJoint,axis=0));
    # compute the posterior prob
    logSPost = logSJoint - logSMarginal;
    SPostLog = numpy.exp(logSPost);
    # find the predicted label
    predClassLog = SPostLog.argmax(0);

    err = (predClassLog!=LTE).sum() / LTE.size;
    print("Accuracy :")
    print(err);
    print("Checks:")
    print((numpy.load("Solution/logSJoint_MVG.npy")-logSJoint).sum());
    print((numpy.load("Solution/logMarginal_MVG.npy")-logSMarginal).sum());
    print((numpy.load("Solution/logPosterior_MVG.npy")-logSPost).sum());
    print();

def NaiveBayes_Classifier(DTR,LTR,DTE,LTE) :
    # since the number of features is small, we can adapt the MVG code by simply zeroing the out-of-diagonal elements of the MVG ML solution. This can be done, for example, multiplying element-wise the MVG ML solution with the identity matrix. The rest of the code remains unchanged. If we have large dimensional data, it may be advisable to implement ad-hoc functions to work directly with just the diagonal of the covariance matrices
    # Implemented !!!! See Cfast
    print("NaiveBayes-Classifier");
    ## -------------- MODEL ---------------------
    u = []; # array of means vectors by class
    C = []; # array of covariance matrices by class
    for i in range(3) :
        # take the samples of class i
        DTRi = DTR[:,LTR==i];
        # compute the mean
        u.append(mcol(DTRi.mean(1)));
        # center the points
        DTRiC = DTRi - u[i];
        # compute the covariance matrix with MVG metod
        C.append(numpy.dot(DTRiC,DTRiC.T)/DTRiC.shape[1]);
        C[i] = C[i] * numpy.identity(DTRiC.shape[0]); # take only the diagonal
        # compute the covariance matrix fast method
        Cfast = numpy.diag((DTRiC**2).sum(1))/DTRiC.shape[1];
    ## --------------- INFERENCE ---------------
    SJoint = []; # array of joint probability by class then vetical Stacked
    PrioP = [1/3,1/3,1/3]; # Prior prob of classes
    for i in range(3) :
        # compute the likelihoods function
        S = mrow(numpy.exp(logpdf_GAU_ND(DTE,u[i],C[i])));
        # compute the join distribution
        SJoint.append(S*PrioP[i]);
    # vertically stack the SJoint array into numpy.ndarray
    SJoint = numpy.vstack(SJoint);
    # compute the marginal denisty
    SMarginal = mrow(SJoint.sum(0));
    # compute the posterior prob
    SPost = SJoint/SMarginal;
    # find the predicted label
    predClass = SPost.argmax(0);

    ## ------------ ACCURACY OF MODEL ------------
    # (num Wrong prediction)/(num of samples)
    err = (predClass!=LTE).sum() / LTE.size;
    print("Accuracy :")
    print(err);
    print("Checks:")
    print((numpy.load("Solution/SJoint_NaiveBayes.npy")-SJoint).sum());
    print((numpy.load("Solution/Posterior_NaiveBayes.npy")-SPost).sum());

    ## -------------  AGAIN BUT WITH LOG ---------
    print("Log-densities:")
    logSJoint = [];
    for i in range(3) :
        S = mrow(logpdf_GAU_ND(DTE,u[i],C[i]));
        logSJoint.append(S+numpy.log(PrioP[i]));
    logSJoint = numpy.vstack(logSJoint);
    # caclulate the log-sum-exp trick
    logSMarginal = mrow(scipy.special.logsumexp(logSJoint,axis=0));
    # compute the posterior prob
    logSPost = logSJoint - logSMarginal;
    SPostLog = numpy.exp(logSPost);
    # find the predicted label
    predClassLog = SPostLog.argmax(0);

    err = (predClassLog!=LTE).sum() / LTE.size;
    print("Accuracy :")
    print(err);
    print("Checks:")
    print((numpy.load("Solution/logSJoint_NaiveBayes.npy")-logSJoint).sum());
    print((numpy.load("Solution/logMarginal_NaiveBayes.npy")-logSMarginal).sum());
    print((numpy.load("Solution/logPosterior_NaiveBayes.npy")-logSPost).sum());
    print();

def Tied_MVG_Classifier(DTR,LTR,DTE,LTE) :
    print("Tied Covariance Gaussian Classifier");
    ## --------------- MODEL ------------------
    u = []; # array of means vectors by class
    C = numpy.zeros((DTR.shape[0],DTR.shape[0])); # covariance matrix
    for i in range(3) :
        # take the samples of class i
        DTRi = DTR[:,LTR==i];
        # compute the mean
        u.append(mcol(DTRi.mean(1)));
        # center the points
        DTRiC = DTRi - u[i];
        # compute the partial covariance matrix and add it to within-class
        C += numpy.dot(DTRiC,DTRiC.T);
    # divide the partial covariance by the number of samples
    C /= DTR.shape[1];

    ## --------------- INFERENCE ---------------
    SJoint = []; # array of joint probability by class then vetical Stacked
    PrioP = [1/3,1/3,1/3]; # Prior prob of classes
    for i in range(3) :
        # compute the likelihoods function
        S = mrow(numpy.exp(logpdf_GAU_ND(DTE,u[i],C)));
        # compute the join distribution
        SJoint.append(S*PrioP[i]);
    # vertically stack the SJoint array into numpy.ndarray
    SJoint = numpy.vstack(SJoint);
    # compute the marginal denisty
    SMarginal = mrow(SJoint.sum(0));
    # compute the posterior prob
    SPost = SJoint/SMarginal;
    # find the predicted label
    predClass = SPost.argmax(0);

    ## ------------ ACCURACY OF MODEL ------------
    # (num Wrong prediction)/(num of samples)
    err = (predClass!=LTE).sum() / LTE.size;
    print("Accuracy :")
    print(err);
    print("Checks:")
    print((numpy.load("Solution/SJoint_TiedMVG.npy")-SJoint).sum());
    print((numpy.load("Solution/Posterior_TiedMVG.npy")-SPost).sum());

    ## -------------  AGAIN BUT WITH LOG ---------
    print("Log-densities:")
    logSJoint = [];
    for i in range(3) :
        S = mrow(logpdf_GAU_ND(DTE,u[i],C));
        logSJoint.append(S+numpy.log(PrioP[i]));
    logSJoint = numpy.vstack(logSJoint);
    # caclulate the log-sum-exp trick
    logSMarginal = mrow(scipy.special.logsumexp(logSJoint,axis=0));
    # compute the posterior prob
    logSPost = logSJoint - logSMarginal;
    SPostLog = numpy.exp(logSPost);
    # find the predicted label
    predClassLog = SPostLog.argmax(0);

    err = (predClassLog!=LTE).sum() / LTE.size;
    print("Accuracy :")
    print(err);
    print("Checks:")
    print((numpy.load("Solution/logSJoint_TiedMVG.npy")-logSJoint).sum());
    print((numpy.load("Solution/logMarginal_TiedMVG.npy")-logSMarginal).sum());
    print((numpy.load("Solution/logPosterior_TiedMVG.npy")-logSPost).sum());
    print();

def Tied_NaiveBayes_Classifier(DTR,LTR,DTE,LTE) :
    print("Tied Naive Bayes Classifier");
    ## ---------------- MODEL ----------------
    u = []; # array of means vectors by class
    C = numpy.zeros((DTR.shape[0],DTR.shape[0])); # covariance matrix
    for i in range(3) :
        # take the samples of class i
        DTRi = DTR[:,LTR==i];
        # compute the mean
        u.append(mcol(DTRi.mean(1)));
        # center the points
        DTRiC = DTRi - u[i];
        # compute the partial covariance matrix and add it to within-class
        C += numpy.dot(DTRiC,DTRiC.T);
        C = C * numpy.identity(DTRiC.shape[0]); # take only the diagonal
        # compute the partial covariance matrix fast method
        Cfast = numpy.diag((DTRiC**2).sum(1))
    # divide the partial covariance by the number of samples
    C /= DTR.shape[1];

    ## --------------- INFERENCE ---------------
    SJoint = []; # array of joint probability by class then vetical Stacked
    PrioP = [1/3,1/3,1/3]; # Prior prob of classes
    for i in range(3) :
        # compute the likelihoods function
        S = mrow(numpy.exp(logpdf_GAU_ND(DTE,u[i],C)));
        # compute the join distribution
        SJoint.append(S*PrioP[i]);
    # vertically stack the SJoint array into numpy.ndarray
    SJoint = numpy.vstack(SJoint);
    # compute the marginal denisty
    SMarginal = mrow(SJoint.sum(0));
    # compute the posterior prob
    SPost = SJoint/SMarginal;
    # find the predicted label
    predClass = SPost.argmax(0);

    ## ------------ ACCURACY OF MODEL ------------
    # (num Wrong prediction)/(num of samples)
    err = (predClass!=LTE).sum() / LTE.size;
    print("Accuracy :")
    print(err);
    print("Checks:")
    print((numpy.load("Solution/SJoint_TiedNaiveBayes.npy")-SJoint).sum());
    print((numpy.load("Solution/Posterior_TiedNaiveBayes.npy")-SPost).sum());

    ## -------------  AGAIN BUT WITH LOG ---------
    print("Log-densities:")
    logSJoint = [];
    for i in range(3) :
        S = mrow(logpdf_GAU_ND(DTE,u[i],C));
        logSJoint.append(S+numpy.log(PrioP[i]));
    logSJoint = numpy.vstack(logSJoint);
    # caclulate the log-sum-exp trick
    logSMarginal = mrow(scipy.special.logsumexp(logSJoint,axis=0));
    # compute the posterior prob
    logSPost = logSJoint - logSMarginal;
    SPostLog = numpy.exp(logSPost);
    # find the predicted label
    predClassLog = SPostLog.argmax(0);

    err = (predClassLog!=LTE).sum() / LTE.size;
    print("Accuracy :")
    print(err);
    print("Checks:")
    print((numpy.load("Solution/logSJoint_TiedNaiveBayes.npy")-logSJoint).sum());
    print((numpy.load("Solution/logMarginal_TiedNaiveBayes.npy")-logSMarginal).sum());
    print((numpy.load("Solution/logPosterior_TiedNaiveBayes.npy")-logSPost).sum());
    print();

def MVG_Classifier_Model(DTR, LTR, nK) :
    u = []; # array of means vectors by class
    C = []; # array of covariance matrices by class
    ## MULTIVARIATE GAUSSIAN CLASSIFIER
    for i in numpy.arange(nK) : 
        ## ESTIMATION OF MODEL
        # only the class i
        DTRi = DTR[:, LTR==i];
        # compute the mean
        u.append(mcol(DTRi.mean(1)));
        # center the points
        DTRiC = DTRi - u[i];
        # compute the covariance matrix
        C.append(numpy.dot(DTRiC,DTRiC.T)/DTRiC.shape[1]);
    return u, C;

def NaiveBayes_Classifier_Model(DTR, LTR, nK) :
    u = []; # array of means vectors by class
    C = []; # array of covariance matrices by class
    for i in range(nK) :
        # take the samples of class i
        DTRi = DTR[:,LTR==i];
        # compute the mean
        u.append(mcol(DTRi.mean(1)));
        # center the points
        DTRiC = DTRi - u[i];
        # compute the covariance matrix fast method
        C.append(numpy.diag((DTRiC**2).sum(1))/DTRiC.shape[1]);
    return u, C;

def Tied_MVG_Classifier_Model(DTR, LTR, nK) :
    u = []; # array of means vectors by class
    D = DTR.shape[0]; # Dimensions of the dataset
    C = numpy.zeros((D,D)); # covariance matrix inizialization
    for i in range(nK) :
        # take the samples of class i
        DTRi = DTR[:,LTR==i];
        # compute the mean
        u.append(mcol(DTRi.mean(1)));
        # center the points
        DTRiC = DTRi - u[i];
        # compute the partial covariance matrix and add it to within-class
        C += numpy.dot(DTRiC,DTRiC.T);
    # divide the partial covariance by the number of samples
    C /= DTR.shape[1];
    return u, C;

def Tied_NaiveBayes_Classifier_Model(DTR,LTR,nK) :
    u = []; # array of means vectors by class
    D = DTR.shape[0]; # Dimensions of the dataset
    C = numpy.zeros((D,D)); # covariance matrix
    for i in range(nK) :
        # take the samples of class i
        DTRi = DTR[:,LTR==i];
        # compute the mean
        u.append(mcol(DTRi.mean(1)));
        # center the points
        DTRiC = DTRi - u[i];
        # compute the partial covariance matrix fast method
        C += numpy.diag((DTRiC**2).sum(1))
    # divide the partial covariance by the number of samples
    C /= DTR.shape[1];
    return u, C;

def inference(DTE, nK, u, C, prioP, fullCov) :
    logSJoint = []; # array of joint probability by class then vetical Stacked
    for i in numpy.arange(nK) :
        # compute the likelihoods function
        if fullCov :
            S = mrow(logpdf_GAU_ND(DTE,u[i],C[i]));
        else :
            S = mrow(logpdf_GAU_ND(DTE,u[i],C));
        # compute the join distribution
        logSJoint.append(S+numpy.log(prioP[i]));
    # vertically stack the SJoint array into numpy.ndarray
    logSJoint = numpy.vstack(logSJoint);
    # caclulate the log-sum-exp trick
    logSMarginal = mrow(scipy.special.logsumexp(logSJoint,axis=0));
    # compute the posterior prob
    logSPost = logSJoint - logSMarginal;
    SPostLog = numpy.exp(logSPost);

    return logSJoint, SPostLog;

def KFold_CrossValidation(D, L, K, prioP, seed = 0) :
    if K < 2 :
        print("Minimum K = 2");
        return
    i = 2;
    Keff = 0;
    # find the maximum possible K-fold, truncate by defect
    N = D.shape[1]; # number of samples in the dataset
    nK = numpy.unique(L).size; # number of classes in the dataset
    while i <= K :
        if N%i == 0 :
            Keff = i
        i += 1;
    sizeFold = int(N/Keff); # dimension of a K-fold sub-dataset
    # set the seed
    numpy.random.seed(seed);
    # create a vector (,1) of random number no repetitions
    idx = numpy.random.permutation(D.shape[1]); # randomize the samples
    SPost = [] ; # array of SPost matrices  one for each Model  
    SPost = [numpy.zeros((nK,N)) for i in range(4)];
    logSJoint = [] ; # array of SJoint matrices one for each Model  
    logSJoint = [numpy.zeros((nK,N)) for i in range(4)];
    for i in range(Keff) :
        # divide the random numbers in Keff-fold parts
        idxTest = idx[(i*sizeFold):((i+1)*sizeFold)];
        idxTrain = numpy.append(idx[:(i*sizeFold)], idx[((i+1)*sizeFold):]);
        # print("FOLD",i);print("Test");print(idxTest);print("Training");print(idxTrain);print();
        DTR = D[:,idxTrain];
        LTR = L[idxTrain];
        DTE = D[:,idxTest];
        LTE = L[idxTest];

        u, C = MVG_Classifier_Model(DTR,LTR,nK);
        logSJoint[0][:,idxTest], SPost[0][:,idxTest] = inference(DTE, nK, u, C, prioP, True);
        # praticamente SPost è un array python composto da 4 numpy.arrays, inference ritorna una matrice (#Class,#SampleInDTE), per metterli nella giusta posizione in cui li abbiamo pescati randomicamente usiamo [:,idxTest] con idxTest = [posizione in cui li abbiamo presi]

        u, C = NaiveBayes_Classifier_Model(DTR,LTR,nK);
        logSJoint[1][:,idxTest], SPost[1][:,idxTest] = inference(DTE, nK, u, C, prioP, True);

        u, C = Tied_MVG_Classifier_Model(DTR,LTR,nK);
        logSJoint[2][:,idxTest], SPost[2][:,idxTest] = inference(DTE, nK, u, C, prioP, False);

        u, C = Tied_NaiveBayes_Classifier_Model(DTR,LTR,nK);
        logSJoint[3][:,idxTest], SPost[3][:,idxTest] = inference(DTE, nK, u, C, prioP, False);

    ## --------- EVALUATION ------------   
    for i in range(4) :
        pred = SPost[i].argmax(0);
        acc = (pred!=L).sum() / N; # calculate the accuracy of model
        print(acc);

    ## ----------- CHECK SOLUTIONS ---------
    print();
    print("Checks:")
    sol = numpy.load("Solution/LOO_logSJoint_MVG.npy");
    print((sol - logSJoint[0]).sum());

    sol = numpy.load("Solution/LOO_logSJoint_NaiveBayes.npy");
    print((sol - logSJoint[1]).sum());

    sol = numpy.load("Solution/LOO_logSJoint_TiedMVG.npy");
    print((sol - logSJoint[2]).sum());

    sol = numpy.load("Solution/LOO_logSJoint_TiedNaiveBayes.npy");
    print((sol - logSJoint[3]).sum());

if __name__ == "__main__" :
    # take data
    D, L = loadIrisDataset();
    # divide in training and test set
    (DTR, LTR),(DTE, LTE) = split_db_2to1(D, L, 0);
    # MVG Classifier
    MVG_Classifier(DTR,LTR,DTE,LTE);
    # Naive Bayes Classifier
    NaiveBayes_Classifier(DTR,LTR,DTE,LTE);
    # Tied Covariance Gaussian Classifier
    Tied_MVG_Classifier(DTR,LTR,DTE,LTE);
    # Tied Naive Bayes Classifier
    Tied_NaiveBayes_Classifier(DTR,LTR,DTE,LTE);
    # K-fold Cross Validation
    KFold_CrossValidation(D,L,D.shape[1],[1/3,1/3,1/3]);

        
