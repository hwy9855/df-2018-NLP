import pandas as pd
import numpy as np
import re
import jieba


def hasNumbers(inputString):
    return bool(re.search(r'[a-z]+', inputString)) or \
           bool(re.search(r'[A-Z]+', inputString)) or \
           bool(re.search(r'[0-9]+', inputString)) or \
           bool(re.search(r'[!*.*,*·*（*）*/*～*…*~*#*=*\-*(*)*—*]+', inputString)) or \
           bool(re.search(r'\s', inputString))


def get_idx(vocab, word):
    for i in range(len(vocab)):
        if vocab[i] == word:
            return i
    return -1

def doc2vec(docs, vocab):
    d = len(docs)
    w = len(vocab)
    content_vec = np.zeros([d, w])
    i = 0
    for doc in docs:
        for word in doc:
            if word in vocab:
                content_vec[i][get_idx(vocab, word)] += 1
        i += 1
    return content_vec

def tfidf(content_vec):
    d = content_vec.shape[0]
    w = content_vec.shape[1]
    tf = np.zeros([d, w])
    idf = np.zeros(w)
    tf_idf = np.zeros([d, w])

    # calculate term frequency
    for i in range(d):
        ni = sum(content_vec[i][:])
        for j in range(w):
            tf[i][j] = content_vec[i][j] / ni

    # calculate inverse document frequency
    for i in range(w):
        D = 0
        for j in range(d):
            if content_vec[j][i] > 0:
                D += 1
        idf[i] = np.log(d / (D + 1))

    # calculate tf-idf
    for i in range(d):
        for j in range(w):
            tf_idf[i][j] = tf[i][j] * idf[j]
    return tf_idf

if __name__ == '__main__':
    # 添加关键词
    jieba.load_userdict('../data/sentiment_words.txt')
    # 读入数据
    data = pd.read_csv('../data/train.csv')

    # high_w_txt = open('../data/sentiment_words.txt')
    # high_w = high_w_txt.readlines()
    # sentiment_words = list()
    # sentiment_id = list()
    # for line in high_w:
    #     sentiment_words.append(line.split('\n')[0])

    content = data['content']

    stop_words_file = open('../data/stop_words.txt')
    stop_words = list()
    lines = stop_words_file.readlines()
    for line in lines:
        stop_words.append(line.split('\n')[0])

    content_list = list()
    for i in range(content.count()):
        content_list.append(list(jieba.cut(content[i])))
    a = dict()
    for i in range(content.count()):
        for w in content_list[i]:
            if w not in stop_words and not hasNumbers(w):
                if w not in a:
                    a[w] = 1
                else:
                    a[w] += 1

    tmp = list()
    for i in a:
        if a[i] < 10:
            tmp.append(i)

    for i in tmp:
        del a[i]

    vocab = list()

    # s = 0
    for w in a:
        vocab.append(w)
        # if w in sentiment_words:
        #     sentiment_id.append(s)
        # s += 1
    # print(s)

    # high_weight_txt = open('../high_weight.txt', 'w')
    # for i in sentiment_id:
    #     high_weight_txt.write(str(i))
    #     high_weight_txt.write('\n')

    vocab_txt = open('../vocab_withoutD.txt', 'w')
    vec_txt = open('../content_vec_withoutD.csv', 'w')
    #tfidf_txt = open('../tf_idf.csv', 'w')
    content_vec = doc2vec(content_list, vocab)
    #tf_idf = tfidf(content_vec)
    for i in range(content_vec.shape[0]):
        for j in range(content_vec.shape[1]):
            if j > 0:
                vec_txt.write(',')
                #tfidf_txt.write(',')
            if content_vec[i][j] == 0:
                vec_txt.write(str(0))
                #tfidf_txt.write(str(0))
            else:
                vec_txt.write(str(content_vec[i][j]))
                #tfidf_txt.write(str(tf_idf[i][j]))
        vec_txt.write('\n')
        #tfidf_txt.write('\n')
        print(i)
    for i in vocab:
        vocab_txt.write(str(i))
        vocab_txt.write('\n')