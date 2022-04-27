import sys;
import math;

N=7;

class Room:
    def __init__(self, N) :
        self.matrix = {};
        self.dim = N;
    
    def addLightSpots(self, x, y) :
        for dist in range(self.dim) :
            startx = x - dist;
            starty = y - dist;
            currVal = 1.0/2**dist;
            #print("startx : ",startx,"starty : ",starty);
            for i in range(dist*2+1) :
                for j in range(dist*2+1) :
                    currx = startx + j;
                    curry = starty + i;
                    #print("currx : ",currx,"curry : ",curry);
                    if currx>=0 and curry>=0 and currx<self.dim and curry<self.dim and not (startx<currx<startx+dist*2 and starty<curry<starty+dist*2) :
                        point = (currx,curry);
                        if point not in self.matrix :
                            self.matrix[point] = 0;
                        self.matrix[point] += math.floor(currVal*10)/10;
                        #print(currx, curry);
            #self.printRoom();
    
    def printRoom(self) :
        for i in range(self.dim) :
            for j in range(self.dim) :
                print("%.1f" % self.matrix[(i,j)] if (i,j) in self.matrix else 0.0 ,end=" ");
            print(); 
        print();   
        
if __name__ == "__main__" :
    # r = Room(7);
    # r1 = Room(7);
    # r2 = Room(7);
    # r.addLightSpots(0,0)
    # r1.addLightSpots(2,3);
    # r2.addLightSpots(4,3);
    # r.printRoom();
    # r1.printRoom();
    # r2.printRoom();

    r = Room(7);
    with open(sys.argv[1]) as f:
        for line in f:
            fields = line.split();
            r.addLightSpots(float(fields[0]),float(fields[1]));
    r.printRoom();
    