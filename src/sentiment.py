from sklearn import svm
import lightgbm as lgb
import numpy as np
import pandas as pd

from bdc import Bdc
from doc2vec import Doc2Vec
from getResult import GetResult

if __name__ == '__main__':
    res = pd.read_csv('../tmp/baseline.csv')
    train = pd.read_csv('../data/train.csv')
    train_vec = pd.read_csv('../content_vec_withoutD.csv', header=None)
    train_vec = np.array(train_vec)
    test_vec = Doc2Vec.test2vec()

    value_list = list(train['sentiment_value'])


    subject_vocab = list(['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观'])

    subject_list = list()
    for i in train['subject']:
        for k in range(10):
            if subject_vocab[k] == i:
                subject_list.append(k)
                break

    predict_subject = list()
    for i in res['subject']:
        for k in range(10):
            if subject_vocab[k] == i:
                predict_subject.append(k)
                break

    predict_value = np.zeros(len(predict_subject))

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

        bdc = Bdc.cal_bdc(train_vec_with_subject, value_list_with_subject, 3)

        for k in range(train_vec_with_subject.shape[0]):
            for j in range(train_vec_with_subject.shape[1]):
                if train_vec_with_subject[k][j] != 0:
                    train_vec_with_subject[k][j] = bdc[j] * train_vec_with_subject[k][j]
                # if j in high_weight_list:
                #     train_vec_with_subject[k][j] = 3 * train_vec_with_subject[k][j]

        for k in range(test_vec_with_subject.shape[0]):
            for j in range(test_vec_with_subject.shape[1]):
                if test_vec_with_subject[k][j] != 0:
                    test_vec_with_subject[k][j] = bdc[j] * test_vec_with_subject[k][j]
                # if j in high_weight_list:
                #     test_vec_with_subject[k][j] = 3 * test_vec_with_subject[k][j]

        # clf = svm.LinearSVR()
        # clf.fit(train_vec_with_subject, value_list_with_subject)

        clf = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.06, n_estimators=200, reg_lambda=1, reg_alpha=0.1)

        clf.fit(train_vec_with_subject, value_list_with_subject)

        print(test_vec_with_subject.shape)
        predict_value_i = clf.predict(test_vec_with_subject)

        k = 0
        for d in range(len(predict_subject)):
            if i == predict_subject[d]:
                predict_value[d] = predict_value_i[k]
                k += 1

    zeros = 0
    ones = 0
    minus = 0

    for i in range(len(predict_value)):
        print(predict_value[i])
        if predict_value[i] > 0.4:
            predict_value[i] = 1
            ones += 1
        elif predict_value[i] < -0.4:
            predict_value[i] = -1
            minus += 1
        else:
            predict_value[i] = 0
            zeros += 1

    print(zeros, ones, minus)
    GetResult.res2doc_sentiment(list(res['content_id']), predict_subject, predict_value)