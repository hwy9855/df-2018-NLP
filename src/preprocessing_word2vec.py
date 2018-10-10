from gensim import corpora
from gensim.models import word2vec
import numpy as np
import pandas as pd

if __name__ == '__main__':

    model = word2vec.Word2Vec.load('../tmp/word2vec.model')
    corpus = corpora.MmCorpus('../tmp/auto.mm')
    dictionary = corpora.Dictionary.load('../tmp/auto.dict')


    content_vec = list()
    for sentence in corpus:
        decvec = np.zeros(200)
        for word in sentence:
            if dictionary[word[0]] in model:
                decvec += model[dictionary[word[0]]]
        content_vec.append(decvec)

    pd.DataFrame.to_csv(pd.DataFrame(np.array(content_vec)), '../tmp/word2vec.csv')
