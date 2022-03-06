import sys
import math

class Point : 
    def __init__(self, coords) :
        self.coords = coords;
    
    def dist(self, other) :
        sum = 0;
        # check the type and dimension
        if type(other) is not Point or len(self.coords) != len(other.coords):
            print(self, end=" ");
            print("and",end=" ");
            print(other,end=" ");
            print("cannot be added");
            return 0;
        # sum
        for i in range(len(self.coords)) : 
            sum += (self.coords[i]-other.coords[i])**2;
        # square root
        return math.sqrt(sum);

    def __str__(self) :
        return str(self.coords);

if __name__ == "__main__" :
    flag = sys.argv[2][1];
    par = sys.argv[3];
    result = 0.0;
    with open(sys.argv[1]) as f :
        prevP = None;
        for line in f:
            fields = line.split();
            busId = fields[0];
            lineId = fields[1];
            coords = (float(fields[2]),float(fields[3]));
            if flag == "b" and busId == par :
                if prevP == None :
                    prevP = Point(coords);
                    currP = Point(coords);
                else :
                    prevP = currP;
                    currP = Point(coords);
                #print(prevP);
                #print(currP);
                #print();
                result += prevP.dist(currP);
    if flag == "b" :
        print("%s - Total Distance: %.1f" % (par,result));