from sklearn import svm
import numpy as np
import pandas as pd
import jieba

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

def test2vec():
    # 添加关键词
    jieba.load_userdict('../data/dict.txt')
    # 读入数据
    data = pd.read_csv('../data/test_public.csv')
    vocab_txt = open('../vocab.txt')
    vocab = list()
    lines = vocab_txt.readlines()
    for line in lines:
        vocab.append(line.split('\n')[0])
    content = data['content']
    content_list = list()
    for i in range(content.count()):
        content_list.append(list(jieba.cut(content[i])))
    test_vec = doc2vec(content_list, vocab)
    return test_vec

def test2vec_sentiment():
    # 添加关键词
    jieba.load_userdict('../data/sentiment_words.txt')
    # 读入数据
    data = pd.read_csv('../data/test_public.csv')
    vocab_txt = open('../vocab_sentiment.txt')
    vocab = list()
    lines = vocab_txt.readlines()
    for line in lines:
        vocab.append(line.split('\n')[0])
    content = data['content']
    content_list = list()
    for i in range(content.count()):
        content_list.append(list(jieba.cut(content[i])))
    test_vec = doc2vec(content_list, vocab)
    return test_vec

def calculate_fc(set, label):
    rows = set.shape[0]
    csum = set.sum(axis=1)
    fc = dict()
    for i in range(rows):
        currentlabel = label[i]
        if not currentlabel in fc:
            fc[currentlabel] = 0
        fc[currentlabel] += csum[i]
    #print('fc succeed')
    #print(fc)
    return fc

def calculate_ftc(fc, set, label, K):
    cols = set.shape[1]
    rows = set.shape[0]
    ftc = np.matrix(np.zeros([K,cols]))
    counti = 0
    for i in fc:
        #print(i)
        for j in range(cols):
            for k in range(rows):
                if label[k] == i:
                    ftc[counti,j] += set[k,j]
        counti += 1
    #print('ftc succeed')
    return ftc

def calculate_ptc(ftc, fcv, set, K):
    cols = set.shape[1]
    ptc = ftc.copy()
    for i in range(K):
        for j in range(cols):
            ptc[i,j] = ftc[i,j] / fcv[i]
    #print('ptc succeed')
    return ptc

def calculate_bdc(ptc, set, K):
    cols = set.shape[1]
    bdc = np.zeros(cols)
    #print(bdc)
    for i in range(cols):
        tmp = 0
        for j in range(K):
            tttmp = 0
            for k in range(K):
                tttmp += ptc[k,i]
            if not tttmp == 0:
                ttmp = ptc[j,i] / tttmp
                if not ttmp == 0:
                    tmp += ttmp * np.log(ttmp)
        bdc[i] = 1 + tmp / 3
    print('bdc succeed')
    return bdc


def cal_subject(train_vec, test_vec, subject_list):
    fc = calculate_fc(train_vec, subject_list)
    fcv = list(fc.values())
    ftc = calculate_ftc(fc, train_vec, subject_list, 10)
    ptc = calculate_ptc(ftc, fcv, train_vec, 10)
    bdc = calculate_bdc(ptc, train_vec, 10)

    train_vec_tmp = train_vec.copy()

    for i in range(train_vec_tmp.shape[0]):
        for j in range(train_vec_tmp.shape[1]):
            if train_vec_tmp[i][j] != 0:
                train_vec_tmp[i][j] = bdc[j] * train_vec[i][j]

    for i in range(test_vec.shape[0]):
        for j in range(test_vec.shape[1]):
            if test_vec[i][j] != 0:
                test_vec[i][j] = bdc[j] * test_vec[i][j]

    clf = svm.LinearSVC()
    clf.fit(train_vec_tmp, subject_list)
    predict_subject = clf.predict(test_vec)
    return predict_subject

def cal_value(train_vec, test_vec, subject_list, predict_subject, value_list):
    predict_value = np.zeros(len(predict_subject))

    high_weight_list = list()
    high_weight = open('../high_weight.txt')
    hw = high_weight.readlines()
    for line in hw:
        high_weight_list.append(int(line.split('\n')[0]))

    for i in range(10):
        train_i = 0
        test_i = 0
        for d in subject_list:
            if d == i:
                train_i += 1

        for d in predict_subject:
            if d == i:
                test_i += 1

        train_vec_with_subject = np.zeros([train_i, train_vec.shape[1]])
        value_list_with_subject = np.zeros([train_i])
        test_vec_with_subject = np.zeros([test_i, test_vec.shape[1]])

        k = 0
        for d in range(train_vec.shape[0]):
            if subject_list[d] == i:
                train_vec_with_subject[k] = train_vec[d]
                value_list_with_subject[k] = value_list[d]
                k += 1

        k = 0
        for d in range(test_vec.shape[0]):
            if predict_subject[d] == i:
                test_vec_with_subject[k] = test_vec[d]
                k += 1

        fc = calculate_fc(train_vec_with_subject, value_list_with_subject)
        fcv = list(fc.values())
        ftc = calculate_ftc(fc, train_vec_with_subject, value_list_with_subject, 3)
        ptc = calculate_ptc(ftc, fcv, train_vec_with_subject, 3)
        bdc = calculate_bdc(ptc, train_vec_with_subject, 3)

        for k in range(train_vec_with_subject.shape[0]):
            for j in range(train_vec_with_subject.shape[1]):
                if train_vec_with_subject[k][j] != 0:
                    train_vec_with_subject[k][j] = bdc[j] * train_vec_with_subject[k][j]
                if j in high_weight_list:
                    train_vec_with_subject[k][j] = 3 * train_vec_with_subject[k][j]

        for k in range(test_vec_with_subject.shape[0]):
            for j in range(test_vec_with_subject.shape[1]):
                if test_vec_with_subject[k][j] != 0:
                    test_vec_with_subject[k][j] = bdc[j] * test_vec_with_subject[k][j]
                if j in high_weight_list:
                    test_vec_with_subject[k][j] = 3 * test_vec_with_subject[k][j]

        clf = svm.LinearSVC()
        clf.fit(train_vec_with_subject, value_list_with_subject)
        predict_value_i = clf.predict(test_vec_with_subject)

        k = 0
        for d in range(len(predict_subject)):
            if i == predict_subject[d]:
                predict_value[d] = predict_value_i[k]
                k += 1

    return predict_value

def res2file(predict_subject, predict_value, subject_vocab):
    res = open('../res.csv', 'w')
    data = pd.read_csv('../data/test_public.csv')
    content_id = data['content_id']
    i = 0
    res.write('content_id,subject,sentiment_value,sentiment_word\n')
    for id in content_id:
        res.write(id)
        res.write(',')
        res.write(subject_vocab[predict_subject[i]])
        res.write(',')
        res.write(str(int(predict_value[i])))
        res.write(',\n')
        i += 1

def get_res():
    train_vec = pd.read_csv('../content_vec.csv', header=None)
    train_vec_sentiment = pd.read_csv('../content_vec_sentiment.csv', header=None)
    train_vec = np.array(train_vec)
    train_vec_sentiment = np.array(train_vec_sentiment)
    data = pd.read_csv('../data/train.csv')
    subject_vocab = list()
    for i in data['subject']:
        if i not in subject_vocab:
            subject_vocab.append(i)
    subject_list = list()
    for i in data['subject']:
        for k in range(10):
            if subject_vocab[k] == i:
                subject_list.append(k)
                break
    value_list = list()
    for i in data['sentiment_value']:
        value_list.append(i)

    '''fc = calculate_fc(train_vec[:9447], subject_list[:9447])
    fcv = list(fc.values())
    ftc = calculate_ftc(fc, train_vec[:9447], subject_list[:9447])
    ptc = calculate_ptc(ftc, fcv, train_vec[:9447])
    bdc = calculate_bdc(ptc, train_vec[:9447])

    for i in range(train_vec.shape[0]):
        for j in range(train_vec.shape[1]):
            if train_vec[i][j] != 0:
                train_vec[i][j] = bdc[j] * train_vec[i][j]'''

    test_vec = test2vec()
    test_vec_sentiment = test2vec_sentiment()

    predict_subject = cal_subject(train_vec, test_vec, subject_list)
    predict_value = cal_value(train_vec_sentiment, test_vec_sentiment, subject_list, predict_subject, value_list)
    res2file(predict_subject, predict_value, subject_vocab)

def testF1():
    train_vec = pd.read_csv('../content_vec.csv', header=None)
    train_vec_sentiment = pd.read_csv('../content_vec_sentiment.csv', header=None)
    train_vec = np.array(train_vec)
    train_vec_sentiment = np.array(train_vec_sentiment)

    data = pd.read_csv('../data/train.csv')
    subject_vocab = list()
    for i in data['subject']:
        if i not in subject_vocab:
            subject_vocab.append(i)

    subject_list = list()
    for i in data['subject']:
        for k in range(10):
            if subject_vocab[k] == i:
                subject_list.append(k)
                break

    value_list = list()
    for i in data['sentiment_value']:
        value_list.append(i)

    predict_subject = cal_subject(train_vec[:9447], train_vec[9447:], subject_list[:9447])
    predict_value = cal_value(train_vec_sentiment[:9447], train_vec_sentiment[9447:], subject_list[:9447], predict_subject, value_list[:9447])

    subject_label = subject_list[9447:]
    value_label = value_list[9447:]


    k = 0
    Ts = 0
    Tv = 0
    T = 0
    for i in range(len(subject_label)):
        if predict_subject[i] == subject_label[i]:
            Ts += 1
        if value_label[i] == predict_value[i]:
            Tv += 1
        if predict_subject[i] == subject_label[i] and value_label[i] == predict_value[i]:
            T += 1
        k += 1
    F1s = (2 * (Ts / k)) / (Ts / k + 1)
    F1v = (2 * (Tv / k)) / (Tv / k + 1)
    F1 = (2 * (T / k)) / (T / k + 1)
    print (k, ',', Ts, ',', F1s, ',', Tv, ',', F1v, ',', T, ',', F1)

if __name__ == '__main__':
    # testF1()
    get_res()

