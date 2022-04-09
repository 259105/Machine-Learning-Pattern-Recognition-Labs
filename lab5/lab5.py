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


if __name__ == "__main__" :
    # take data
    D, L = loadIrisDataset();
    # divide in training and test set
    (DTR, LTR),(DTE, LTE) = split_db_2to1(D, L, 0);

    u = []; # means
    C = []; # Covariance matrixes
    S = []; # Score matrix
    PrioP = [1/3, 1/3, 1/3] # Prior prob
    SJoint = [] ;
    ## MULTIVARIATE GAUSSIAN CLASSIFIER
    for i in [0,1,2] : 
        ## ESTIMATION OF MODEL
        # only the class i
        DTRi = DTR[:, LTR==i];
        # compute the mean
        u.append(mcol(DTRi.mean(1)));
        DTRiC = DTRi - u[i];
        # compute the covariance matrix
        C.append(numpy.dot(DTRiC,DTRiC.T)/DTRiC.shape[1]);
        # print(u[i]); print(C[i]);

        ## INFERENCE FOR TEST SAMPLE
        # compute the likelihoods function
        S.append(mrow(numpy.exp(logpdf_GAU_ND(DTE,u[i],C[i]))));
        # compute the join distribution
        SJoint.append(S[i]*PrioP[i]);

    # Check solution 
    #sol = numpy.load('Solution/SJoint_MVG.npy');
    #print(numpy.vstack(SJoint).shape);
    #print(sol.shape);
    #print((numpy.vstack(SJoint) - sol).sum());

    SJoint = numpy.vstack(SJoint);
    # compute the marginal denisty
    margDens = mrow(SJoint.sum(0));
    # compute the posterior prob
    SPost = SJoint/margDens;
    # find the predicted label
    predClass = SPost.argmax(0);
    #print(predClass); print(LTE);

    ## ACCURACY OF MODEL
    # (num Wrong prediction)/(num of samples)
    err = (predClass!=LTE).sum() / LTE.size;
    print(err);

    ## AGAIN BUT WITH LOG
    S = [];
    logSJoint = [];
    for i in [0,1,2] :
        S.append(mrow(logpdf_GAU_ND(DTE,u[i],C[i])));
        logSJoint.append(S[i]+numpy.log(PrioP[i]));
    logSJoint = numpy.vstack(logSJoint);
    # caclulate the log-sum-exp trick
    logSMarginal = mrow(scipy.special.logsumexp(logSJoint,axis=0));
    logSPost = logSJoint - logSMarginal;
    SPostLog = numpy.exp(logSPost);

    predClassLog = SPostLog.argmax(0);

    err = (predClass!=LTE).sum() / LTE.size;
    print(err);
    # print((numpy.load("Solution/logSJoint_MVG.npy")-logSJoint).sum());
    # print((numpy.load("Solution/logMarginal_MVG.npy")-logSMarginal).sum());
    # print((numpy.load("Solution/logPosterior_MVG.npy")-logSPost).sum());

        
