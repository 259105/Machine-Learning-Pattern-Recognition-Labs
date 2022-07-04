from cmath import pi
from traceback import print_tb
import numpy;
import matplotlib.pyplot as plt;

def mrow(v):
    return v.reshape((1,v.size));

def mcol(v):
    return v.reshape((v.size,1));

def logpdf_GAU_ND(x, mu, C) :
    first = -mu.shape[0]/2*numpy.log(2*numpy.pi);
    second = -1/2*numpy.linalg.slogdet(C)[1];
    xc = x - mu; # center the values
    third = -1/2*numpy.dot(numpy.dot(xc.T,numpy.linalg.inv(C)),xc);
    # take only the rows (i,i) 
    return numpy.diagonal(first+second+third);

def loglikelihood(x,u,C) :
    return logpdf_GAU_ND(x,u,C).sum(0);

if __name__ == "__main__" : 
    XND = numpy.linspace(-8,12,1000);
    mu = numpy.ones((1,1)) * 1.0;
    C = numpy.ones((1,1)) * 2.0;

    ## CHECK SOLUTION ##
    pdfSol = numpy.load('./llGAU.npy')
    pdfGau = logpdf_GAU_ND(mrow(XND), mu, C)
    print(numpy.abs(pdfSol - pdfGau).max())

    plt.figure();
    plt.plot(XND.ravel(), numpy.exp(logpdf_GAU_ND(mrow(XND), mu, C)));
    plt.show();

    ## CHECK SOLUTION ##
    XND = numpy.load('./XND.npy')
    mu = numpy.load('./muND.npy')
    C = numpy.load('./CND.npy')
    pdfSol = numpy.load('./llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print("My error from solution: ",numpy.abs(pdfSol - pdfGau).max())
    
    plt.figure();
    plt.plot(XND.ravel(), numpy.exp(logpdf_GAU_ND(mrow(XND), mu, C)));
    plt.show(); 

    ## MAXIMUM LIKELIHOOD ESTIMATE
    # compute the mean
    uML = mcol(XND.mean(1));
    # center the samples
    XNDC = XND - uML;
    # compute the variance matrix
    vML = numpy.dot(XNDC,XNDC.T)/XNDC.shape[1];
    print(uML); print(vML);
    # print log-likelihood
    print(loglikelihood(XND,uML,vML));

    ## SAME, BUT DIFFERENT INPUT
    X1D = numpy.load("./X1D.npy");
    m_ML = mcol(X1D.mean(1));
    X1DC = X1D - m_ML;
    C_ML = numpy.dot(X1DC,X1DC.T)/X1DC.shape[1];
    print(m_ML); print(C_ML);

    ## PLOT DIFF BETWEEN THE REAL DATA AND MY ESTIMATION
    plt.figure();
    plt.hist(X1D.ravel(), bins=50, density=True); # real data plot
    XPlot = numpy.linspace(-8, 12, 1000);
    plt.plot(XPlot.ravel(),numpy.exp(logpdf_GAU_ND(mrow(XPlot), m_ML, C_ML))); 
    plt.show();
    # estimated density plot
    print(loglikelihood(X1D,m_ML,C_ML));
