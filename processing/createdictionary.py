from collections import defaultdict
import os
import dill
def main():
    directory = "/Users/elliotschumacher/Dropbox/git/10-807-Project-Group/hu-en/"
    alignfn = "europarl-v7.hu-en.comb_train_tok.gdfa"
    combfn = "europarl-v7.hu-en.comb_train_tok"
    lang1 = "en"
    lang2 = "hu"


    onetwodict = defaultdict(lambda: defaultdict(int))
    twoonedict = defaultdict(lambda: defaultdict(int))

    with open(os.path.join(directory, alignfn)) as alignf, open(os.path.join(directory, combfn)) as combf:
        for align_line, comb_line in zip(alignf, combf):
            lsplit = comb_line.split("|||")
            l1 = lsplit[0].strip().split(" ")
            l2 = lsplit[1].strip().split(" ")
            alignments = align_line.strip().split(" ")
            alignmentpairs = [x.split("-") for x in alignments]

            for pair in alignmentpairs:
                l1ind = int(pair[0])
                l2ind = int(pair[1])

                onetwodict[l1[l1ind]][l2[l2ind]] += 1
                twoonedict[l2[l2ind]][l1[l1ind]] += 1

    with open(os.path.join(directory, "dict_{0}_{1}.dill".format(lang1, lang2)), "wb") as l1l2, open(os.path.join(directory, "dict_{0}_{1}.dill".format(lang2, lang1)), "wb") as l2l1:
        dill.dump(onetwodict, file=l1l2)
        dill.dump(twoonedict, file=l2l1)
    with open(os.path.join(directory, "dict_{0}_{1}.txt".format(lang1, lang2)), "w") as l1l2, open(os.path.join(directory, "dict_{0}_{1}.txt".format(lang2, lang1)), "w") as l2l1:
        for word1 in sorted(onetwodict):
            line = word1 +  " | "
            for word2 in sorted(onetwodict[word1], key=onetwodict[word1].get, reverse=True):
                line += " {0}:{1}".format(word2, onetwodict[word1][word2])
            l1l2.write(line + "\n")

        for word1 in sorted(twoonedict):
            line = word1 + " | "
            for word2 in sorted(twoonedict[word1], key=twoonedict[word1].get, reverse=True):
                line += " {0}:{1}".format(word2, twoonedict[word1][word2])
            l2l1.write(line + "\n")

    with open(os.path.join(directory, "dict1to1_{0}_{1}.txt".format(lang1, lang2)), "w") as l1l2, open(os.path.join(directory, "dict1to1_{0}_{1}.txt".format(lang2, lang1)), "w") as l2l1:
        for word1 in sorted(onetwodict):
            line = word1 + " ||| "
            word2 = sorted(onetwodict[word1], key=onetwodict[word1].get, reverse=True)[0]
            line += word2
            l1l2.write(line + "\n")

        for word1 in sorted(twoonedict):
            line = word1 + " ||| "
            word2 = sorted(twoonedict[word1], key=twoonedict[word1].get, reverse=True)[0]
            line += word2
            l2l1.write(line + "\n")
    pass

if __name__ == '__main__':
    main()
