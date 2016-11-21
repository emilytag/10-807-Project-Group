

from random import shuffle

def main():
    l1filename = "/Users/elliotschumacher/Downloads/hu-en/europarl-v7.hu-en.en"
    l2filename = "/Users/elliotschumacher/Downloads/hu-en/europarl-v7.hu-en.hu"
    pairs = []
    emptylines = 0
    with open(l1filename) as l1file, open(l2filename) as l2file:
        for line1, line2 in zip(l1file, l2file):
            if line1.strip() == "" and line2.strip() == "":
                emptylines += 1
            else:

                pairs.append((line1, line2))
    shuffle(pairs)
    numlines = len(pairs)
    train = int(float(numlines) * 0.7)
    dev = train + int(float(numlines) * 0.15)
    test = dev + int(float(numlines) * 0.15)

    trainlines = pairs[:train]
    devlines = pairs[train:dev]
    testlines = pairs[dev:]

    print(emptylines)

    with open(l1filename + "_train", "w") as l1train, open(l2filename + "_train", "w") as l2train:
        for l1, l2 in trainlines:
            l1train.write(l1)
            l2train.write(l2)

    with open(l1filename + "_dev", "w") as l1train, open(l2filename + "_dev", "w") as l2train:
        for l1, l2 in devlines:
            l1train.write(l1)
            l2train.write(l2)

    with open(l1filename + "_test", "w") as l1train, open(l2filename + "_test", "w") as l2train:
        for l1, l2 in testlines:
            l1train.write(l1)
            l2train.write(l2)




if __name__ == '__main__':
    main()
