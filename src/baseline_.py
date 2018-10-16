import pandas as pd
import numpy as np
from bdc import Bdc
from lgb import Lgb
from doc2vec import Doc2Vec
from getResult import GetResult

def get_res(iter, baseline):
    train_vec = pd.read_csv('../content_vec_withoutD.csv', header=None)
    test_file = pd.read_csv('../data/test_public.csv')

    # train_vec_sentiment = pd.read_csv('../content_vec_sentiment.csv', header=None)
    train_vec = np.array(train_vec)
    # train_vec_sentiment = np.array(train_vec_sentiment)
    data = pd.read_csv('../data/train.csv')
    subject_vocab = list(['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观'])

    subject_list = list()
    for i in data['subject']:
        for k in range(10):
            if subject_vocab[k] == i:
                subject_list.append(k)
                break

    value_list = list()
    for i in data['sentiment_value']:
        value_list.append(i)

    bdc = Bdc.cal_bdc(train_vec, subject_list, 10)
    for i in range(train_vec.shape[0]):
        for j in range(train_vec.shape[1]):
            if train_vec[i][j] > 0:
                train_vec[i][j] = bdc[j]

    print(train_vec)
    test_vec = Doc2Vec.test2vec()
    for i in range(test_vec.shape[0]):
        for j in range(test_vec.shape[1]):
            if test_vec[i][j] > 0:
                test_vec[i][j] = bdc[j]

    print(test_vec)
    test_id = list(test_file['content_id'])

    res_id, res_subject, value_list = Lgb.cal_subject_mul(train_vec, subject_list, test_id, test_vec, iter, baseline)

    GetResult.res2doc_mul(res_id, res_subject, value_list)

if __name__ == '__main__':
    baseline = pd.read_csv('~/submit.csv')
    get_res(100, baseline)
