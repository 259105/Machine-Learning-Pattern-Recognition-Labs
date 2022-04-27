import sys

class Book :
    def __init__(self, isbn):
        self.isbn = isbn;
        self.boughtWeightedPrices = 0.0;
        self.soldWeightedPrices = 0.0
        self.soldCopies = 0;
        self.boughtCopies = 0;
    
    def availableCopies(self):
        return self.boughtCopies-self.soldCopies;
    
    def sold(self, numberOfCopies, price) :
        self.soldCopies += numberOfCopies;
        self.soldWeightedPrices += numberOfCopies*price;
    
    def bought(self, numberfOfCopies, price) :
        self.boughtCopies += numberfOfCopies;
        self.boughtWeightedPrices += numberfOfCopies*price;
    
    def gainPerBook(self) :
        avgBought = self.boughtWeightedPrices/self.boughtCopies;
        avgSold = self.soldWeightedPrices/self.soldCopies;
        return avgSold-avgBought;
    
if __name__=="__main__" :
    books = {};
    booksSoldByMonth = {};
    with open(sys.argv[1]) as f:
        for line in f :
            fields = line.split();
            book = fields[0];
            op = fields[1];
            month = fields[2][3:];
            numberOfCopies = int(fields[3]);
            price = float(fields[4]);
            # if not exist create a new entry
            if book not in books :
                books[book] = Book(book);
            # add sold or bought
            if op == "B" :
                books[book].bought(numberOfCopies,price);
            elif op =="S" :
                books[book].sold(numberOfCopies,price);
                # add to month dictionary
                if month not in booksSoldByMonth :
                    booksSoldByMonth[month] = 0;
                booksSoldByMonth[month] += numberOfCopies;
    
    print("Available Copies:")
    for book in books :
        print("\t%s: " % book, books[book].availableCopies());
    print("Sold books per month:")
    for month in booksSoldByMonth :
        print("\t%s" % month, booksSoldByMonth[month]);
    print("Gain per book:")
    for book in books :
        bookI = books[book];
        gpb = bookI.gainPerBook();
        soldCopies = bookI.soldCopies;
        gain = gpb*soldCopies;
        print("\t%s: %.1f (avg %.1f, sold %d)" % (book, gain, gpb, soldCopies));
