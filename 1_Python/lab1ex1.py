import sys

class Competitor:
    def __init__(self,name,surname,country,judges):
        self.name = name;
        self.surname = surname;
        self.country = country;
        self.judges = [float(i) for i in judges];
    
    def score(self):
        return sum(sorted(self.judges)[1:-1]);

if __name__ == "__main__" :
    top3Competitor = [];
    countryScore = {};
    with open(sys.argv[1]) as f:
        for line in f:
            fields = line.split();
            name, surname, country = fields[:3];
            judges = fields[3:];
            competitor = Competitor(name,surname,country,judges);
            #Competitor(fields[0],fields[1],fields[2],fields[3:])
            top3Competitor.append(competitor);
            top3Competitor = sorted(top3Competitor, key = lambda i: i.score())[::-1][0:3]
            if country not in countryScore :
                countryScore[country] = 0;
            countryScore[country] += competitor.score();

    if len(countryScore) == 0 : 
        print("No competitors");
        sys.exit(0);
    
    print("Final ranking:");
    for pos, comp in enumerate(top3Competitor) :
        print("%d: %s %s - Score: %.1f" % (pos+1, comp.name, comp.surname, comp.score()));
    print();
    print("Best country:");
    bestCountry = None;
    for country in countryScore:
        if  bestCountry is None or countryScore[country]>countryScore[bestCountry] :
            bestCountry = country;
    print("%s Total Score: %.1f" % (bestCountry, countryScore[bestCountry]));


