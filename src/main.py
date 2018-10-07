from sklearn import svm
import lightgbm as lgb
import numpy as np
import pandas as pd

from bdc import Bdc
from doc2vec import Doc2Vec
from lgb import Lgb
from getResult import GetResult


def cal_subject(train_vec, test_vec, subject_list):
    bdc = Bdc.cal_bdc(train_vec, subject_list, 10)

    train_vec_tmp = train_vec.copy()

    for i in range(train_vec_tmp.shape[0]):
        for j in range(train_vec_tmp.shape[1]):
            if train_vec_tmp[i][j] != 0:
                train_vec_tmp[i][j] = bdc[j] * train_vec[i][j]

    for i in range(test_vec.shape[0]):
        for j in range(test_vec.shape[1]):
            if test_vec[i][j] != 0:
                test_vec[i][j] = bdc[j] * test_vec[i][j]

    # clf = svm.LinearSVC()
    clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=55, reg_alpha=0.0, reg_lambda=1,
        max_depth=15, n_estimators=6000, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)

    clf.fit(train_vec_tmp, subject_list)
    predict_subject = clf.predict(test_vec)
    print(predict_subject)
    return predict_subject

def cal_value(train_vec, test_vec, subject_list, predict_subject, value_list):
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

        # clf = svm.LinearSVC()
        clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=55, reg_alpha=0.0, reg_lambda=1,
                                 max_depth=15, n_estimators=6000, objective='binary',
                                 subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                                 learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4,
                                 min_child_samples=5)
        clf.fit(train_vec_with_subject, value_list_with_subject)
        print(test_vec_with_subject.shape)
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
    test_file = pd.read_csv('../data/test_public.csv')

    # train_vec_sentiment = pd.read_csv('../content_vec_sentiment.csv', header=None)
    train_vec = np.array(train_vec)
    # train_vec_sentiment = np.array(train_vec_sentiment)
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

    test_vec = Doc2Vec.test2vec()
    test_id = list(test_file['content_id'])

    res_id, res_subject = Lgb.cal_label(train_vec, subject_list, test_id, test_vec)

    GetResult.res2doc(res_id, res_subject)

    # test_vec_sentiment = test2vec_sentiment()
    #
    # predict_subject = cal_subject(train_vec, test_vec, subject_list)
    # # predict_value = cal_value(train_vec, test_vec, subject_list, predict_subject, value_list)
    # res2file(predict_subject, np.zeros(test_vec.shape[0]), subject_vocab)

def testF1():
    train_vec = pd.read_csv('../content_vec.csv', header=None)
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

    test_id_csv = data['content_id']
    test_id = list(test_id_csv)
    test_id = test_id[9447:]
    test_vec = train_vec[9447:]
    test_id_single = list()
    test_vec_single = list()
    for l in range(len(test_id)):
        if test_id[l] not in test_id_single:
            test_id_single.append(test_id[l])
            test_vec_single.append(test_vec[l])

    res_id, res_subject = Lgb.cal_label(train_vec[:9447], subject_list[:9447], test_id_single, test_vec_single)
    GetResult.cal_F1(res_id, res_subject, 9447)

    '''
    predict_subject = cal_subject(train_vec[:9447], train_vec[9447:], subject_list[:9447])
    # predict_value = cal_value(train_vec[:9447], train_vec[9447:], subject_list[:9447], predict_subject, value_list[:9447])

    subject_label = subject_list[9447:]
    value_label = value_list[9447:]


    k = 0
    Ts = 0
    Tv = 0
    T = 0
    for i in range(len(subject_label)):
        if predict_subject[i] == subject_label[i]:
            Ts += 1
        if value_label[i] == 0:
            Tv += 1
        if predict_subject[i] == subject_label[i] and value_label[i] == 0:
            T += 1
        k += 1
    F1s = (2 * (Ts / k)) / (Ts / k + 1)
    F1v = (2 * (Tv / k)) / (Tv / k + 1)
    F1 = (2 * (T / k)) / (T / k + 1)
    print (k, ',', Ts, ',', F1s, ',', Tv, ',', F1v, ',', T, ',', F1)
    '''

if __name__ == '__main__':
    # testF1()
    get_res()

