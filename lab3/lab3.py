import numpy
import matplotlib.pyplot as plt
import sys

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
    plt.show();

if __name__ == "__main__" :
    m = int(sys.argv[1]);

    ## GETTING THE DATA ##
    D, L = loadIrisDataset();
    # Labels: Iris-setosa, Iris-versicolor, Iris-virginica
    # Attrs: sepal-length, sepal-width, petal-length, petal-width 
    #print(D);
    #print(L);

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
    DP = numpy.dot(P.T,D);

    ## PLOTTING THE DATA ##
    #scatter2attrs(DP,L);
