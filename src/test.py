from gensim import corpora
from gensim.models import word2vec
import re
import jieba
import pandas as pd


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


if __name__ == '__main__':
    jieba.load_userdict('../data/dict.txt')
    data = pd.read_csv('../data/train.csv')
    contents = data['content']
    stop_txt = open('../data/stop_words.txt', 'r')
    stop_lines = stop_txt.readlines()
    stop_set = list()
    for line in stop_lines:
        stop_set.append(line.split('\n')[0])
    content_sentences = list()
    for sentence in contents:
        cut_sentence = list(jieba.cut(sentence))
        dele = list()
        for word in cut_sentence:
            if str(word) in stop_set:
                dele.append(word)
            elif hasNumbers(word):
                dele.append(word)
        for word in dele:
            cut_sentence.remove(word)
        content_sentences.append(cut_sentence)

    frequency = dict()
    for sentence in content_sentences:
        for word in sentence:
            if word not in frequency:
                frequency[word] = 1
            else:
                frequency[word] += 1

    words = [[word for word in sentence if frequency[word] > 1]
             for sentence in content_sentences]

    dictionary = corpora.Dictionary(words)
    corpus = [dictionary.doc2bow(word) for word in words]
    vec = word2vec.Word2Vec(words, size=200)
    print(vec, dictionary)

    dictionary.save('../tmp/auto.dict')
    corpora.MmCorpus.serialize('../tmp/auto.mm', corpus)
    vec.save('../tmp/word2vec.model')
