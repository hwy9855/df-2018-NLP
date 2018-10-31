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
        clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=30, reg_alpha=0.0, reg_lambda=1,
                                 max_depth=6, n_estimators=6000, objective='binary',
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

def get_res(iter):
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

    res_id, res_subject = Lgb.cal_subject(train_vec, subject_list, test_id, test_vec, iter)

    GetResult.res2doc(res_id, res_subject)

    # test_vec_sentiment = test2vec_sentiment()
    #
    # predict_subject = cal_subject(train_vec, test_vec, subject_list)
    # # predict_value = cal_value(train_vec, test_vec, subject_list, predict_subject, value_list)
    # res2file(predict_subject, np.zeros(test_vec.shape[0]), subject_vocab)

def testF1(iter):
    train_vec = pd.read_csv('../content_vec_withoutD.csv', header=None)
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

    # value_list = list()
    # for i in data['sentiment_value']:
    #     value_list.append(i)
    train_vec = Bdc.cal_bdc_with_vec(train_vec, subject_list, 10)

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
    for i in range(100):
        print((i + 1) * iter)
        res_id, res_subject = Lgb.cal_subject(train_vec[:9447], subject_list[:9447], test_id_single, test_vec_single, (i + 1) * iter)
        GetResult.cal_F1(res_id, res_subject, 9447)

def testF1_word2vec(iter):
    data = pd.read_csv('../data/train.csv')
    subject_vocab = list(['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观'])

    subject_list = list()
    for i in data['subject']:
        for k in range(10):
            if subject_vocab[k] == i:
                subject_list.append(k)
                break

    train_vec = pd.read_csv('../tmp/word2vec.csv')
    train_vec = np.array(train_vec)

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

    res_id, res_subject = Lgb.cal_subject(train_vec[:9447], subject_list[:9447], test_id_single, test_vec_single, iter)
    GetResult.cal_F1(res_id, res_subject, 9447)

def cvtest():
    res = open('../res.txt1', 'w')
    params = { 'boosting_type':'gbdt', 'num_leaves':55, 'reg_alpha':0.1, 'reg_lambda':0,
              'max_depth':15, 'objective':'binary',
              'subsample':0.8, 'colsample_bytree':0.8, 'subsample_freq':1,
              'learning_rate':0.06, 'min_child_weight':1, 'random_state':20, 'n_jobs':4}

    train_vec = pd.read_csv('../content_vec_withoutD.csv', header=None)
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
    print(train_vec)
    train_vec = Bdc.cal_bdc_with_vec(train_vec, subject_list, 10)
    print(train_vec)

    test_res = list()
    for l in range(len(subject_list)):
        test_res.append(list())
    for i in range(10):
        train_label_onehot = subject_list.copy()
        for l in range(len(subject_list)):
            if subject_list[l] != i:
                train_label_onehot[l] = 0
            else:
                train_label_onehot[l] = 1
        # print(train_label_onehot)
        # print(train_subject)
        data_train = lgb.Dataset(train_vec, train_label_onehot)
        clf = lgb.cv(
            params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
            early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
        res.write(str(len(clf['rmse-mean'])))
        res.write(' ')
        res.write(str(clf['rmse-mean'][-1]))
        res.write('\n')
    data_train = lgb.Dataset(train_vec, subject_list)
    clf = lgb.cv(
        params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
        early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
    res.write(str(len(clf['rmse-mean'])))



if __name__ == '__main__':
    # i = input('0 for test\n1 for run\n')
    # if i == '0':
    #     for iter in range(50):
    #         print((iter + 1) * 100)
    #         testF1((iter + 1) * 100)
    #     # testF1(7500)
    #     cvtest()
    #
    # else:
    #     get_res(int(i))
    # testF1(10)
    # get_res(480)
    cvtest()
    # for i in range(50):
    #     print((i+1)*200)
    #     testF1_word2vec((i+1)*200)