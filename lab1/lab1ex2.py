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

class BusStop :
    def __init__(self, bus, coords, time) :
        self.bus = bus;
        self.coords = coords;
        self.time = time;

if __name__ == "__main__" :
    try :
        flag = sys.argv[2][1];
    except :
        raise KeyError();
    par = sys.argv[3];
    result = 0.0;
    lBussesStopLine = [];
    with open(sys.argv[1]) as f :
        prevP = None;
        for line in f:
            fields = line.split();
            busId = fields[0];
            lineId = fields[1];
            time = float(fields[4]);
            coords = (float(fields[2]),float(fields[3]));
            if flag == "b" and busId == par :
                if prevP == None :
                    prevP = Point(coords);
                    currP = Point(coords);
                else :
                    prevP = currP;
                    currP = Point(coords);
                result += prevP.dist(currP);
            elif flag == "l" and lineId == par :
                lBussesStopLine.append(BusStop(busId,coords,time));

    if flag == "b" :
        print("%s - Total Distance: %.1f" % (par,result));
    elif flag == "l" : 
        distinctBus = set([busStop.bus for busStop in lBussesStopLine])
        orderedStops = sorted(lBussesStopLine, key = lambda b : b.time);
        dist = 0.0;
        time = 0.0;
        for bus in distinctBus :
            prevP = None;
            prevT = 0;
            for busStop in orderedStops :
                if busStop.bus == bus :
                    if prevP == None :
                        prevP = Point(busStop.coords);
                        currP = Point(busStop.coords);
                        prevT = busStop.time;
                        currT = busStop.time;
                    else :
                        prevP = currP;
                        prevT = currT;
                        currP = Point(busStop.coords);
                        currT = busStop.time;
                    
                    dist += prevP.dist(currP);
                    time += currT - prevT;
        avgSpeed = dist/time;
        print("%s - Avg Speed: " % par, avgSpeed);


