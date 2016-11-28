
from nltk import tokenize
from random import shuffle

def main():

    for t in ["dev", "train", "test"]:
        l1filename = "/Users/elliotschumacher/Dropbox/git/10-807-Project-Group/hu-en/europarl-v7.hu-en.en_{0}_tok".format(t)
        l2filename = "/Users/elliotschumacher/Dropbox/git/10-807-Project-Group/hu-en/europarl-v7.hu-en.hu_{0}_tok".format(t)
        loutput = "/Users/elliotschumacher/Dropbox/git/10-807-Project-Group/hu-en/europarl-v7.hu-en.comb_{0}_tok".format(t)
        with open(l1filename) as l1file, open(l2filename) as l2file, open(loutput, "w") as loutf:
            emptylines = 0

            for line1, line2 in zip(l1file, l2file):
                line1 = line1.lower()
                line2 = line2.lower()
                if line1.strip() == "" or line2.strip() == "":
                    emptylines += 1
                else:
                    loutf.write("{0} ||| {1}\n".format(line1.rstrip(), line2.rstrip()))
            print(emptylines)



if __name__ == '__main__':
    main()
