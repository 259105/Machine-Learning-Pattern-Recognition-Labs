import sys;

N=7;

class Room:
    def __init__(self, N) :
        self.matrix = {};
        self.dim = N;
    
    def addLightSpots(self, x, y) :
        for dist in range(self.dim)[::-1] :
            startx = x - dist;
            starty = y - dist;
            currVal = 1.0/2**dist;
            #print("startx : ",startx,"starty : ",starty);
            for i in range(dist*2+1) :
                for j in range(dist*2+1) :
                    currx = startx + j;
                    curry = starty + i;
                    #print("currx : ",currx,"curry : ",curry);
                    if currx>=0 and curry>=0 and currx<self.dim and curry<self.dim :
                        point = (currx,curry);
                        if point not in self.matrix :
                            self.matrix[point] = 0;
                        self.matrix[point] = currVal;
                        print(currx, curry);
    
    def printRoom(self) :
        for i in range(self.dim) :
            for j in range(self.dim) :
                print(self.matrix[(i,j)]);
            print("\n");
                




        
if __name__ == "__main__" :
    r = Room(7);
    r.addLightSpots(0,0);
    r.printRoom();

    