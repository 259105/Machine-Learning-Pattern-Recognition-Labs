import sys;
import numpy;
import matplotlib.pyplot as plt;

def toCol(array) :
    return array.reshape((array.size,1));

def toRow1(v) :
    return v.reshape(-1); # v.size

def toRowN(v) :
    return v.reshape((1,v.size));

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
                datasetFlowers.append(toCol(numpy.array(attrs)));
                labels.append(dicFlowers[fields[-1].strip()]);
            except :
                pass;
        return numpy.hstack(datasetFlowers), numpy.array(labels, dtype=numpy.int32);
    
def loadIrisDataset():

    # The dataset is already available in the sklearn library (pay attention that the library represents samples as row vectors, not column vectors - we need to transpose the data matrix)
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def histograms(D,L) :
    M0 = L==0; # mask [True, False, ....]
    D0 = D[:,M0];
    D1 = D[:,L==1];
    D2 = D[:,L==2];

    dicAttr = ["sepal length", "sepal width", "petal lenght", "petal width"];

    for attr in range(4) : 
        plt.figure();
        plt.xlabel(dicAttr[attr]);
        plt.hist(D0[attr,:],bins=10,density=True,alpha=0.4,label="Setosa");
        plt.hist(D1[attr,:],bins=10,density=True,alpha=0.4,label="Versicolor");
        plt.hist(D2[attr,:],bins=10,density=True,alpha=0.4,label="Virginica");
        plt.legend();
        plt.tight_layout()
        #plt.savefig("hists_%s.pdf" % dicAttr[attr]);
    plt.show();
    
def scatter(D,L) :
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

if __name__ == "__main__" :
    #ds, lab = load(sys.argv[1]);
    ds, lab = loadIrisDataset();
    #print(ds);
    #print(lab);
    
    histograms(ds,lab);
    scatter(ds,lab);

    ## Computing mean -- Easy Level
    mean = 0;
    for i in range(ds.shape[1]) : 
        mean += ds[:,i:i+1];
    mean /= float(ds.shape[1]);
    #print(mean);

    ## Computing mean -- medium level
    mean1 = toCol(ds.sum(axis=1)) / float(ds.shape[1]);
    #print(mean1);

    ## Computing mean -- advanced level
    mean2 = ds.mean(1).reshape((ds.shape[0],1)); #instead of toCol here I have used reshape
    print(mean2);

    DC = ds - mean2; # broadcasting
    print(DC);