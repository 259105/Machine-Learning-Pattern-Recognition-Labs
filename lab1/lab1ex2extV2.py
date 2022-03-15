import sys
import numpy

N=7;

class Room :
    def __init__(self,dim):
        self.matrix = numpy.zeros((dim,dim));
        self.dim = dim;
    
    def addLightspots(self,x,y):
        intensity = [0.2, 0.5, 1.0];
        m = numpy.zeros((self.dim,self.dim));
        for int, dist in zip(intensity,range(3)[::-1]) :
            topLeftX = x-dist;
            topLeftY = y-dist;
            if topLeftX<0 : topLeftX=0;
            if topLeftY<0 : topLeftY=0
            m[topLeftX:((x+dist)+1),topLeftY:((y+dist)+1)]=int;
        self.matrix+=m;

    def printRoom(self) :
        print(self.matrix);

if __name__ == "__main__" :
    r = Room(7);
    with open(sys.argv[1]) as f:
        for line in f :
            fields = line.split();
            fields = [int(s) for s in fields];
            r.addLightspots(fields[0],fields[1]);
    r.printRoom();
