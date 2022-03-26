from cmath import pi
import numpy;
import matplotlib.pyplot as plt;

def mrow(v):
    return v.reshape((1,v.size));

def logpdf_GAU_ND(x, mu, C) :
    first = -mu.shape[0]/2*numpy.log(2*numpy.pi);
    second = -1/2*numpy.linalg.slogdet(C)[1];
    xc = x - mu; # center the values
    third = -1/2*numpy.dot(numpy.dot(xc.T,numpy.linalg.inv(C)),xc);
    return numpy.diagonal(first+second+third);

if __name__ == "__main__" : 
    XPlot = numpy.linspace(-8,12,1000);
    mu = numpy.ones((1,1)) * 1.0;
    C = numpy.ones((1,1)) * 2.0;

    ## CHECK SOLUTION ##
    #pdfSol = numpy.load('./llGAU.npy')
    #pdfGau = logpdf_GAU_ND(mrow(XPlot), m, C)
    #print(numpy.abs(pdfSol - pdfGau).max())

    ## CHECK SOLUTION ##
    XPlot = numpy.load('./XND.npy')
    mu = numpy.load('./muND.npy')
    C = numpy.load('./CND.npy')
    pdfSol = numpy.load('./llND.npy')
    pdfGau = logpdf_GAU_ND(XPlot, mu, C)
    print(numpy.abs(pdfSol - pdfGau).max())
    
    plt.figure();
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(mrow(XPlot), mu, C)));
    plt.show(); 
