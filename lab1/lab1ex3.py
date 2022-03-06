import sys;

monthOfYear = {
    1: "Jenuary",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

if __name__ == "__main__" :
    birthsByCity = {};
    birthsByMonth = {};
    with open(sys.argv[1]) as f :
        for line in f :
            fields = line.split();
            city = fields[2];
            month = monthOfYear[int(fields[3].split("/")[1])];
            if city not in birthsByCity :
                birthsByCity[city] = 0;
            if month not in birthsByMonth :
                birthsByMonth[month] = 0;
            birthsByMonth[month] += 1;
            birthsByCity[city] += 1;

    print("Births per city:")
    for city in birthsByCity:
        print("\t%s: %d" % (city, birthsByCity[city]));
    print("Births per month:")
    for month in birthsByMonth:
        print("\t%s: %s" % (month, birthsByMonth[month]));
    print("Average number of births: ",sum(birthsByCity.values())/len(birthsByCity));
    