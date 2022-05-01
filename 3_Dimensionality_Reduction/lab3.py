import numpy
import scipy
import matplotlib.pyplot as plt
import sys
from collections import Counter

def toCol(v) :
    return v.reshape((v.size,1));

def loadIrisDataset() :
    import sklearn.datasets;
    return sklearn.datasets.load_iris()["data"].T, sklearn.datasets.load_iris()["target"];

def scatter4attrs(D,L) :
    dicAttr = ["sepal length", "sepal width", "petal lenght", "petal width"];
    dicFlowers = ["Setosa","Versicolor","Virginica" ]
    for attri in range(4) :
        for attrj in range(4):
            if attri == attrj : continue;
            plt.figure();
            plt.xlabel(dicAttr[attri]);
            plt.ylabel(dicAttr[attrj]);
            for flower in range(3) :
                classData = D[:,L==flower];
                plt.scatter(classData[attri,:],classData[attrj,:],label=dicFlowers[flower]);
            plt.legend;
            plt.tight_layout;
        plt.show();
    
def scatter2attrs(D,L) :
    plt.figure();
    plt.xlabel("PC-1");
    plt.ylabel("PC-2");
    for i, flowerType in enumerate(["Setosa", "Versicolor","Virginica"]) :
        classData = D[:,L==i];
        plt.scatter(classData[0,:],classData[1,:],label=flowerType);
    plt.legend;
    plt.tight_layout;
    #plt.show();

if __name__ == "__main__" :
    # m = int(sys.argv[1]);
    m = 2

    ## GETTING THE DATA ##
    D, L = loadIrisDataset();
    # Labels: Iris-setosa, Iris-versicolor, Iris-virginica
    # Attrs: sepal-length, sepal-width, petal-length, petal-width 
    #print(D);
    #print(L);
    K = len(Counter(L).keys()); # number of distinct elements in the array
                                # it is the number of distinct label
    #print(K);

    ############################## PCA #####################################

    ## CENTERING THE DATA ##
    DC = D - toCol(D.mean(1));
    #print(DC);
    #scatter4attrs(DC,L);

    ## COMPUTING EMPIRICAL COVARIANCE MATRIX ##
    C = (numpy.dot(DC,DC.T))/DC.shape[1];
    #print(C);

    ## COMPUTING EIGVALUES AND EIGVECTORS ##
    s, U = numpy.linalg.eigh(C);
    #print(s.shape);
    #print(U.shape);
    #print(s);
    #print(U);
    
    ## PRINT ERRO OF MY SOLUTION VS THE SOL OF PROF ##
    #print(U-numpy.load("IRIS_PCA_matrix_m4.npy")[:,::-1]);

    ## COMPUTING P (Projection matrix for tranformation)
    P = U[:, ::-1][:,0:m];
    #print(P);

    ## PROJECTING DATA ON NEW BASE ##
    DPPCA = numpy.dot(P.T,D);

    ## PLOTTING THE DATA ##
    scatter2attrs(DPPCA,L);

    ############################## LDA #####################################

    ## COMPUTING Sw (WITHIN COVARIANCE MATRIX) AND Sb ##
    Sw = 0;
    Sb = 0;
    for i in range(K) :
        DCl = D[:,L==i];    # take only samples of the class-i
        DClC = DCl - toCol(DCl.mean(1));    # center the data
        MC = toCol(DCl.mean(1)) - toCol(D.mean(1)); # center the mean of class, respect the global mean
        ## COMPUTING ELEMENT-I OF THE SUMMATORY OF Sb
        Sb += DClC.shape[1]*numpy.dot(MC,MC.T);
        # Swc = numpy.dot(DClC,DClC.T)/DClC.shape[1]; # covariance matrix for class-i
        # Sw +=Swc*DClC.shape[1];
        # in order to save time we can remove div and mul
        ## COMPUTING ELEMENT-I OF THE SUMMATORY OF Sw
        Sw += numpy.dot(DClC,DClC.T);
    Sw = Sw/DC.shape[1];
    Sb = Sb/DC.shape[1];
    #print(Sw);
    #print(Sb);

    ## COMPUTING THE EIG VALUES OF THE GENERALIZED EIGENVALUE PROBLEM FOR HERMITIAN MATRICIES
    s, U = scipy.linalg.eigh(Sb,Sw); # numpy here don't work, numpy don't solve the generalized problem
    W = U[:,::-1][:,0:(m if m<K else K-1)]; # take the voluted dimension if it is <= K-1, constraints given by the LDA
 
    ## PRINT ERRO OF MY SOLUTION VS THE SOL OF PROF ##
    print(W-numpy.load("IRIS_LDA_matrix_m2.npy"));

    ## MADE THE W MATRIX ORTOGONAL ##
    UW= numpy.linalg.svd(W)[0];
    print(UW[1]);
    WO = UW[:,0:(m if m<K else K-1)];
    #print(WO)
    #print(W);

    ## CHECK IF IT'S CORRECT ##
    #print(numpy.linalg.svd(numpy.hstack([W,numpy.load("IRIS_LDA_matrix_m2.npy")]))[1]); # must have at most m non-zero singular values

    ## PROJECTING DATA ON NEW BASE ##
    DPLDAO = numpy.dot(WO.T,D);
    DPLDA = numpy.dot(W.T,D);
    ## PLOTTING THE DATA ##
    scatter2attrs(DPLDAO,L);
    scatter2attrs(DPLDA,L);
    plt.show();
