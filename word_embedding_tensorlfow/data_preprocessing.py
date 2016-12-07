# -*- coding: utf-8 -*-
import nltk
import string
from itertools import izip

def create_train_data(input_file, output_file):
    with open(input_file, "r") as f, open(output_file, "wr") as ff:
        for line in f.readlines():
            # text = u' '.join(nltk.word_tokenize(line.decode("utf-8").lower()))
            text = ' '.join(nltk.word_tokenize(line.decode("utf-8").lower()))
            # text = text.decode('utf8')
            # text = unicode(text, 'utf-8')
            ff.write("%s " % text.encode("utf-8"))


def create_train_data2(input_file, output_file):
    with open(input_file, "r") as f, open(output_file, "wr") as ff:
        for line in f.readlines():
            # text = u' '.join(nltk.word_tokenize(line.decode("utf-8").lower()))

            text = line.lower().translate(None, string.punctuation)
            text = ' '.join(nltk.word_tokenize(text.decode("utf-8")))
            # text = ' '.join([x for x in text.split() if not any(c.isdigit() for c in x)])

            ff.write("%s " % text.encode("utf-8"))


def merge_file(input_file1, input_file2 ,output_file):

    with open(input_file1, "r") as f, open(input_file2, "r") as ff, open(output_file, "wr") as fff:
        for line1, line2 in izip(f.readlines(), ff.readlines()):
            fff.write("%s %s" % (line1.split()[0], line2))


            # fff.write("%s " % text.encode("utf-8"))

merge_file("vocab_uzb.txt", "w2v_table_uzb.txt", "word_embedding_uzb.txt")