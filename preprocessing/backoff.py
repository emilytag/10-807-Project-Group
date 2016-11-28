from collections import defaultdict
import os
import dill
from gensim import models

def backoff(w, model_l1, model_l3, l2l3_dict, l2l1_dict):
    if w in model_l1:
        word1vec = model_l1[w]
        topvecs = model_l3.similar_by_vector(word1vec, topn=50)
        word_backoff = w
        for wordv, sim in topvecs:
            if wordv in l2l3_dict:
                word3vec = model_l3[wordv]
                l1_replacement = model_l1.similar_by_vector(word3vec, topn=5)
                word_backoff = l1_replacement[0][0]
    else:
        word_backoff = w
    return word_backoff

def backoff_l1(w, model_l1, l1l2_dict):
    if w in model_l1:
        word1vec = model_l1[w]
        topvecs = model_l1.similar_by_vector(word1vec, topn=50)
        word_backoff = w
        for wordv, sim in topvecs:
            if wordv in l1l2_dict:
                word_backoff = wordv
    else:
        word_backoff = w
    return word_backoff

def intersect_dicts(dictl1tol2, dictl1tol3):
    l1l2words = set(dictl1tol2.keys())
    l1l3words = set(dictl1tol3.keys())
    l1words_inter = l1l2words.intersection(l1l3words)
    l2l3_dict = {}
    l3l2_dict = {}
    for w1 in l1words_inter:
        maxword_l2 = sorted(dictl1tol2[w1], key=dictl1tol2[w1].get, reverse=True)[0]
        maxword_l3 = sorted(dictl1tol3[w1], key=dictl1tol3[w1].get, reverse=True)[0]
        l2l3_dict[maxword_l2] = maxword_l3
        l3l2_dict[maxword_l3] = maxword_l2
    return l2l3_dict, l3l2_dict



def main():
    directory = "/Users/elliotschumacher/Dropbox/git/10-807-Project-Group/hu-en/"
    directory2 = "/Users/elliotschumacher/Dropbox/git/10-807-Project-Group/fi-en/"

    lang1 = "en"
    l1emb = "/Users/elliotschumacher/Dropbox/git/10-807-Project-Group/word_embedding/word_embedding_en.txt"
    lang2 = "hu"
    l2emb = "/Users/elliotschumacher/Dropbox/git/10-807-Project-Group/word_embedding/word_embedding_hu.txt"
    lang3 = "fi"
    l3emb = "/Users/elliotschumacher/Dropbox/git/10-807-Project-Group/word_embedding/fi2enemb.txt"


    sourcefile = "europarl-v7.hu-en.en_test_tok"


    with open(os.path.join(directory, "dict_{0}_{1}.dill".format(lang1, lang2)), "rb") as l1l2, \
            open(os.path.join(directory, "dict_{0}_{1}.dill".format(lang2, lang1)), "rb") as l2l1, \
            open(os.path.join(directory2, "dict_{0}_{1}.dill".format(lang1, lang3)), "rb") as l1l3, \
            open(os.path.join(directory2, "dict_{0}_{1}.dill".format(lang3, lang1)), "rb") as l3l1:

        dictl1tol2 = dill.load(l1l2)
        model_l1 = models.Word2Vec.load_word2vec_format(l1emb, binary=False)

        dictl2tol1 = dill.load(l2l1)
        model_l2 = models.Word2Vec.load_word2vec_format(l2emb, binary=False)

        model_l3 = models.Word2Vec.load_word2vec_format(l3emb, binary=False)

        dictl1tol3 = dill.load(l1l3)
        dictl3tol1 = dill.load(l3l1)

    l2l3_dict, l3l2_dict = intersect_dicts(dictl1tol2, dictl1tol3)
    cached_words = {}
    missing_count = 0.0
    existing_count = 0.0
    with open(os.path.join(directory, sourcefile)) as sourcef, open(os.path.join(directory, "rep_" + sourcefile), 'w') as sourcefrep:
        for line in sourcef:
            words = [x.strip() for x in line.lower().split(" ")]
            for w_i in range(len(words)):
                w = words[w_i]
                if w in cached_words:
                    w = cached_words[w]
                    missing_count += 1.0

                elif w not in dictl1tol2:
                    missing_count += 1.0
                    #w_rep = backoff(w, model_l1, model_l3,l2l3_dict, dictl2tol1)
                    w_rep = backoff_l1(w, model_l1, dictl1tol2)

                    cached_words[w] = w_rep
                    words[w_i] = w_rep
                    print("O:{0}, R:{1}".format(w, w_rep))
                else:
                    existing_count += 1.0
                words[w_i] = w
            sourcefrep.write(" ".join(words) + "\n")
    print(missing_count)
    print(existing_count)

    print("Missing percentage : {0}".format(missing_count / (missing_count + existing_count)))




    pass

if __name__ == '__main__':
    main()
