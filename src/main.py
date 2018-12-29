import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer,TfidfVectorizer
import lightgbm as lgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,StratifiedKFold
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score,accuracy_score,classification_report
import sklearn.metrics as metrics
from sklearn import svm
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import pickle as pk
from sklearn import svm
from sklearn import neighbors
from dataPreProcessing import *
from bdc import Bdc
from doc2vec import Doc2Vec


def get_data():
    train = pd.read_csv('/home/hujoe/PycharmProjects/df-2018-NLP/data/train.csv')
    test = pd.read_csv('/home/hujoe/PycharmProjects/df-2018-NLP/data/test_public.csv')

    train = train.sample(frac=1)
    train = train.reset_index(drop=True)

    data = pd.concat([train, test])

    lbl = LabelEncoder()
    lbl.fit(train['subject'])
    nb_classes = len(list(lbl.classes_))

    pk.dump(lbl, open('label_encoder.sav', 'wb'))

    subject = lbl.transform(train['subject'])

    y = []
    for i in list(train['sentiment_value']):
        y.append(i)

    y1 = []
    for i in subject:
        y1.append(i)

    return data, train.shape[0], np.array(y).reshape(-1, 1)[:, 0], test['content_id'], np.array(y1).reshape(-1, 1)[:, 0]


def processing_data(data):
    word = jieba.cut(data)
    # words = list(word)
    # stop_words_file = open('/home/hujoe/PycharmProjects/df-2018-NLP/data/stop_words.txt')
    # stop_words = list()
    # lines = stop_words_file.readlines()
    # for line in lines:
    #     stop_words.append(line.split('\n')[0])
    # word = list()
    # for w in words:
    #     if not hasNumbers(w):
    #         word.append(w)
    return ' '.join(word)


def pre_process():
    data, nrw_train, y, test_id, y1 = get_data()

    data['cut_comment'] = data['content'].map(processing_data)
    print(data['cut_comment'].head(5))

    print('TfidfVectorizer')
    tf = TfidfVectorizer(ngram_range=(1, 2), analyzer='char')
    discuss_tf = tf.fit_transform(data['cut_comment'])
    print(discuss_tf)

    print('HashingVectorizer')
    ha = HashingVectorizer(ngram_range=(1, 1), lowercase=False)
    discuss_ha = ha.fit_transform(data['cut_comment'])

    data = hstack((discuss_tf, discuss_ha)).tocsr()
    return data[:nrw_train], data[nrw_train:], y, test_id, y1


def micro_avg_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='micro')


def run():
    X,test,y,test_id,y1= pre_process()
    N = 10
    kf = StratifiedKFold(n_splits=N, random_state=2018).split(X,y)

    # clf = svm.LinearSVC(loss='hinge', tol=1e-4, C=0.6)
    y_train_oofp = np.zeros_like(y, dtype='float64')
    y_train_oofp1 = np.zeros_like(y, dtype='float64')
    '''
    y_train_oofp: 
    y_y_train_oofp1:
    '''

    y_test_oofp = np.zeros((test.shape[0], N))
    y_test_oofp_1 = np.zeros((test.shape[0], N))


    acc = 0
    vcc = 0
    for i, (train_fold, test_fold) in enumerate(kf):
        print(i)
        X_train,          X_validate,     label_train,   label_validate, label_1_train,  label_1_validate, = \
        X[train_fold, :], X[test_fold,:], y[train_fold], y[test_fold],   y1[train_fold], y1[test_fold]

        clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
                                 max_depth=8, n_estimators=15, objective='binary',
                                 subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                                 learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
        clf.fit(X_train, label_1_train)
        val_1 = clf.predict(X_validate)
        # y_train_oofp1[test_fold] = val_

        vcc += micro_avg_f1(label_1_validate, val_1)
        result = clf.predict(test)
        y_test_oofp_1[:, i] = result

    lbl = pk.load(open('label_encoder.sav','rb'))
    res_2 = []
    subject_pred = []
    res = []
    for i in range(y_test_oofp_1.shape[0]):
        tmp = []
        for j in range(N):
            tmp.append(int(y_test_oofp_1[i][j]))
        word_counts = Counter(tmp)
        yes = word_counts.most_common(1)
        res_2.append(lbl.inverse_transform([yes[0][0]])[0])
        subject_pred.append(([yes[0][0]])[0])

    print(subject_pred)
    kf = StratifiedKFold(n_splits=N, random_state=2018).split(X,y)
    for i, (train_fold, test_fold) in enumerate(kf):
        print(i)
        X_train, X_validate, label_train, label_validate, label_1_train, label_1_validate, = \
            X[train_fold, :], X[test_fold, :], y[train_fold], y[test_fold], y1[train_fold], y1[test_fold]

        for subject in range(10):
            clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
                                     max_depth=8, n_estimators=500, objective='binary',
                                     subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                                     learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
            list_train = list()
            list_validate = list()
            list_test = list()
            for l in range(len(label_1_train)):
                if label_1_train[l] == subject:
                    list_train.append(l)
            for l in range(len(label_1_validate)):
                if label_1_validate[l] == subject:
                    list_validate.append(l)
            for l in range(len(subject_pred)):
                if subject_pred[l] == subject:
                    list_test.append(l)

            if len(list_test):

                clf.fit(X_train[list_train], label_train[list_train])

                print(list_test)
                result = clf.predict(test[list_test])
                y_test_oofp[list_test, i] = result

    for i in range(y_test_oofp_1.shape[0]):
        tmp_sentiment = []
        for j in range(N):
            tmp_sentiment.append(y_test_oofp[i][j])
        res.append(max(set(tmp_sentiment), key=tmp_sentiment.count))

    # print(acc / N)
    # print(vcc / N)


    print(res)

    result = pd.DataFrame()
    result['content_id'] = list(test_id)

    result['subject'] = list(res_2)
    result['subject'] = result['subject']

    result['sentiment_value'] = list(res)
    result['sentiment_value'] = result['sentiment_value'].astype(int)

    result['sentiment_word'] = ''
    result.to_csv('submit.csv', index=False)


def cv_test():
    # train_vec = pd.read_csv('/home/hujoe/PycharmProjects/df-2018-NLP/content_vec_withoutD.csv', header=None)
    # test_file = pd.read_csv('/home/hujoe/PycharmProjects/df-2018-NLP/data/test_public.csv')
    #
    # # train_vec_sentiment = pd.read_csv('../content_vec_sentiment.csv', header=None)
    # train_vec = np.array(train_vec)
    # # train_vec_sentiment = np.array(train_vec_sentiment)
    # data = pd.read_csv('/home/hujoe/PycharmProjects/df-2018-NLP/data/train.csv')
    # subject_vocab = list(['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观'])
    #
    # subject_list = list()
    # for i in data['subject']:
    #     for k in range(10):
    #         if subject_vocab[k] == i:
    #             subject_list.append(k)
    #             break
    # subject_list = np.array(subject_list)
    #
    # value_list = list()
    # for i in data['sentiment_value']:
    #     value_list.append(i)
    # value_list = np.array(value_list)
    #
    # bdc = Bdc.cal_bdc(train_vec, subject_list, 10)
    # for i in range(train_vec.shape[0]):
    #     for j in range(train_vec.shape[1]):
    #         if train_vec[i][j] > 0:
    #             train_vec[i][j] = bdc[j]
    #
    # print(train_vec)
    # test_vec = Doc2Vec.test2vec()
    # for i in range(test_vec.shape[0]):
    #     for j in range(test_vec.shape[1]):
    #         if test_vec[i][j] > 0:
    #             test_vec[i][j] = bdc[j]
    # test_id = list(test_file['content_id'])
    X, test, y, test_id, y1 = pre_process()
    # train_vec, test_vec, value_list, test_id, subject_list
    params = { 'boosting_type':'gbdt', 'num_leaves':55, 'reg_alpha':0.1, 'reg_lambda':1,
              'max_depth':8, 'objective':'binary',
              'subsample':0.8, 'colsample_bytree':0.8, 'subsample_freq':1,
              'learning_rate':0.06, 'min_child_weight':1, 'random_state':20, 'n_jobs':4}

    data_train = lgb.Dataset(X, y)
    clf = lgb.cv(
        params, data_train, num_boost_round=10000, nfold=10, stratified=False, shuffle=True, metrics='rmse',
        early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

    print(str(len(clf['rmse-mean'])))

    data_train = lgb.Dataset(X, y1)

    clf = lgb.cv(
        params, data_train, num_boost_round=10000, nfold=10, stratified=False, shuffle=True, metrics='rmse',
        early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

    print(str(len(clf['rmse-mean'])))

def run_single():
    X, test, y, test_id, y1 = pre_process()

    clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
                             max_depth=8, n_estimators=550, objective='binary',
                             subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                             learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)

    y_test_oofp = np.zeros((test.shape[0]))
    y_test_oofp_1 = np.zeros((test.shape[0]))


    clf.fit(X, y)

    val_ = clf.predict(test)
    result = clf.predict(test)
    y_test_oofp[:] = result

    # clf = svm.LinearSVC(loss='hinge', tol=1e-4, C=0.6)

    clf.fit(X, y1)
    result = clf.predict(test)
    y_test_oofp_1[:] = result


    lbl = pk.load(open('label_encoder.sav', 'rb'))
    res_2 = []
    for i in range(y_test_oofp_1.shape[0]):
        res_2.append(lbl.inverse_transform([int(y_test_oofp_1[i])])[0])

    res = []
    for i in range(y_test_oofp.shape[0]):
        res.append(y_test_oofp[i])

    result = pd.DataFrame()
    result['content_id'] = list(test_id)

    result['subject'] = list(res_2)
    result['subject'] = result['subject']

    result['sentiment_value'] = list(res)
    result['sentiment_value'] = result['sentiment_value'].astype(int)

    result['sentiment_word'] = ''
    result.to_csv('submit.csv', index=False)

def run_base():
    X,test,y,test_id,y1= pre_process()
    N = 10
    kf = StratifiedKFold(n_splits=N, random_state=2018).split(X,y)


    clf_0 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=55, reg_alpha=0.1, reg_lambda=1,
                             max_depth=8, n_estimators=500, objective='binary',
                             subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                             learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)

    clf_1 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=55, reg_alpha=0.1, reg_lambda=1,
                             max_depth=8, n_estimators=10, objective='binary',
                             subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                             learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
    y_train_oofp = np.zeros_like(y, dtype='float64')
    y_train_oofp1 = np.zeros_like(y, dtype='float64')
    '''
    y_train_oofp: 
    y_y_train_oofp1:
    '''

    y_test_oofp = np.zeros((test.shape[0], N))
    y_test_oofp_1 = np.zeros((test.shape[0], N))


    acc = 0
    vcc = 0
    # clf_sentiment = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
    #                          max_depth=8, n_estimators=400, objective='binary',
    #                          subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
    #                          learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
    # clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
    #                          max_depth=8, n_estimators=10, objective='binary',
    #                          subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
    #                          learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
    # clf = RandomForestClassifier(n_estimators=20, max_depth=8, max_leaf_nodes=80)

    sub_n = 0
    sem_n = 0
    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_validate, label_train, label_validate, label_1_train, label_1_validate, = X[train_fold, :], X[
                                                                                                               test_fold,
                                                                                                               :], y[
                                                                                                 train_fold], y[
                                                                                                 test_fold], y1[
                                                                                                 train_fold], y1[
                                                                                                 test_fold]
        clf_0.fit(X_train, label_train)

        val_ = clf_0.predict(X_validate)
        y_train_oofp[test_fold] = val_
        # if (micro_avg_f1(label_validate, val_)> 0.73):
        print('sentiment_value_f1:%f' % micro_avg_f1(label_validate, val_))
        sem_n += 1
        acc += micro_avg_f1(label_validate, val_)
        result = clf_0.predict(test)
        y_test_oofp[:, i] = result
        # else:
        #     y_test_oofp[:, i] = i * 100 * np.ones(len(test_id))\
        # clf = svm.LinearSVC(loss='hinge', tol=1e-4, C=0.6)

        clf_1.fit(X_train, label_1_train)
        val_1 = clf_1.predict(X_validate)
        y_train_oofp1[test_fold] = val_
        # if (micro_avg_f1(label_1_validate, val_1) > 0.65):
        print('subject=_f1:%f' % micro_avg_f1(label_1_validate, val_1))
        # sub_n += 1
        vcc += micro_avg_f1(label_1_validate, val_1)
        result = clf_1.predict(test)
        y_test_oofp_1[:, i] = result
        # else:
        #     y_test_oofp_1[:, i] = (i+1) * 100 * np.ones(len(test_id))
        #     print(i * 100 * np.ones(len(test_id)))

    print(acc / N)
    print(vcc / N)

    lbl = pk.load(open('label_encoder.sav', 'rb'))
    res_2 = []
    for i in range(y_test_oofp_1.shape[0]):
        tmp = []
        for j in range(N):
            tmp.append(int(y_test_oofp_1[i][j]))
        word_counts = Counter(tmp)
        yes = word_counts.most_common(1)
        res_2.append(lbl.inverse_transform([yes[0][0]])[0])


    res = []
    for i in range(y_test_oofp.shape[0]):
        tmp = []
        for j in range(N):
            tmp.append(y_test_oofp[i][j])
        res.append(max(set(tmp), key=tmp.count))

    result = pd.DataFrame()
    result['content_id'] = list(test_id)

    result['subject'] = list(res_2)
    result['subject'] = result['subject']

    result['sentiment_value'] = list(res)
    result['sentiment_value'] = result['sentiment_value'].astype(int)

    result['sentiment_word'] = ''
    result.to_csv('submit_without_num.csv', index=False)

def run_base_bdc():
    train_vec = pd.read_csv('/home/hujoe/PycharmProjects/df-2018-NLP/content_vec_withoutD.csv', header=None)
    test_file = pd.read_csv('/home/hujoe/PycharmProjects/df-2018-NLP/data/test_public.csv')

    # train_vec_sentiment = pd.read_csv('../content_vec_sentiment.csv', header=None)
    train_vec = np.array(train_vec)
    # train_vec_sentiment = np.array(train_vec_sentiment)
    data = pd.read_csv('/home/hujoe/PycharmProjects/df-2018-NLP/data/train.csv')
    subject_vocab = list(['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观'])

    subject_list = list()
    for i in data['subject']:
        for k in range(10):
            if subject_vocab[k] == i:
                subject_list.append(k)
                break
    subject_list = np.array(subject_list)

    value_list = list()
    for i in data['sentiment_value']:
        value_list.append(i)
    value_list = np.array(value_list)

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

    N = 10
    kf = StratifiedKFold(n_splits=N, random_state=2018).split(train_vec, subject_list)

    clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
                             max_depth=8, n_estimators=500, objective='binary',
                             subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                             learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
    clf_1 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
                               max_depth=8, n_estimators=10, objective='binary',
                               subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                               learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
    y_train_oofp = np.zeros_like(subject_list, dtype='float64')
    y_train_oofp1 = np.zeros_like(subject_list, dtype='float64')
    '''
    y_train_oofp: 
    y_y_train_oofp1:
    '''

    y_test_oofp = np.zeros((test_vec.shape[0], N))
    y_test_oofp_1 = np.zeros((test_vec.shape[0], N))

    acc = 0
    vcc = 0

    l = 0
    ll = 0
    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_validate, label_train, label_validate, label_1_train, label_1_validate, = \
        train_vec[train_fold, :], train_vec[test_fold,:], value_list[train_fold], value_list[test_fold], subject_list[train_fold], subject_list[test_fold]
        clf.fit(X_train, label_train)

        val_ = clf.predict(X_validate)
        y_train_oofp[test_fold] = val_
        if micro_avg_f1(label_validate , val_) > 0.7:
            l += 1
            print('sentiment_value_f1:%f' % micro_avg_f1(label_validate, val_))
            acc += micro_avg_f1(label_validate, val_)
            result = clf.predict(test_vec)
            y_test_oofp[:, i] = result

        # clf = svm.LinearSVC(loss='hinge', tol=1e-4, C=0.6)

        clf_1.fit(X_train, label_1_train)
        val_1 = clf_1.predict(X_validate)
        y_train_oofp1[test_fold] = val_

        if micro_avg_f1(label_1_validate, val_1) > 0.6:
            ll += 1
            vcc += micro_avg_f1(label_1_validate, val_1)
            result = clf_1.predict(test_vec)
            y_test_oofp_1[:, i] = result

    print(acc / l)
    print(vcc / ll)

    lbl = pk.load(open('../tmp/label_encoder.sav', 'rb'))
    res_2 = []
    for i in range(y_test_oofp_1.shape[0]):
        tmp = []
        for j in range(N):
            tmp.append(int(y_test_oofp_1[i][j]))
        word_counts = Counter(tmp)
        yes = word_counts.most_common(1)
        res_2.append(lbl.inverse_transform([yes[0][0]])[0])

    res = []
    for i in range(y_test_oofp.shape[0]):
        tmp = []
        for j in range(N):
            tmp.append(y_test_oofp[i][j])
        res.append(max(set(tmp), key=tmp.count))

    result = pd.DataFrame()
    result['content_id'] = list(test_id)

    result['subject'] = list(res_2)
    result['subject'] = result['subject']

    result['sentiment_value'] = list(res)
    result['sentiment_value'] = result['sentiment_value'].astype(int)

    result['sentiment_word'] = ''
    result.to_csv('../submit_bdc.csv', index=False)

def run_boost():
    X,test,y,test_id,y1= pre_process()
    N = 10
    kf = StratifiedKFold(n_splits=N, random_state=2018).split(X,y)

    clf_num = 3
    clf_weigh = [10, 3, 2]
    clf_0 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
                             max_depth=8, n_estimators=550, objective='binary',
                             subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                             learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
    clf_1 = svm.LinearSVC(loss='hinge', tol=1e-4, C=0.6)
    clf_2 = linear_model.LogisticRegression(C=1.11)

    y_train_oofp = np.zeros_like(y, dtype='float64')
    y_train_oofp1 = np.zeros_like(y, dtype='float64')

    '''
    y_train_oofp: 
    y_y_train_oofp1:
    '''

    y_test_oofp = np.zeros((test.shape[0], N, clf_num))
    y_test_oofp_1 = np.zeros((test.shape[0], N, clf_num))


    acc = 0
    vcc = 0

    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_validate, label_train, label_validate, label_1_train, label_1_validate, = X[train_fold, :], X[
                                                                                                               test_fold,
                                                                                                               :], y[
                                                                                                 train_fold], y[
                                                                                                 test_fold], y1[
                                                                                                 train_fold], y1[
                                                                                                 test_fold]

        clf_0.fit(X_train, label_train)
        clf_1.fit(X_train, label_train)
        clf_2.fit(X_train, label_train)

        # val_ = clf.predict(X_validate)
        # y_train_oofp[test_fold] = val_
        # print('sentiment_value_f1:%f' % micro_avg_f1(label_validate, val_))
        # acc += micro_avg_f1(label_validate, val_)
        result = clf_0.predict(test)
        y_test_oofp[:, i, 0] = result
        result = clf_1.predict(test)
        y_test_oofp[:, i, 1] = result
        result = clf_2.predict(test)
        y_test_oofp[:, i, 2] = result

        # clf = svm.LinearSVC(loss='hinge', tol=1e-4, C=0.6)

        clf_0.fit(X_train, label_1_train)
        clf_1.fit(X_train, label_1_train)
        clf_2.fit(X_train, label_1_train)
        # val_1 = clf.predict(X_validate)
        # y_train_oofp1[test_fold] = val_
        #
        # vcc += micro_avg_f1(label_1_validate, val_1)

        result = clf_0.predict(test)
        y_test_oofp_1[:, i, 0] = result
        result = clf_1.predict(test)
        y_test_oofp_1[:, i, 1] = result
        result = clf_2.predict(test)
        y_test_oofp_1[:, i, 2] = result

        print(i)
    # print(acc / N)
    # print(vcc / N)

    lbl = pk.load(open('label_encoder.sav', 'rb'))
    res_2 = []
    for i in range(y_test_oofp_1.shape[0]):
        tmp = []
        for j in range(N):
            for clf in range(clf_num):
                for n in range(clf_weigh[clf]):
                    tmp.append(int(y_test_oofp_1[i][j][clf]))
        word_counts = Counter(tmp)
        yes = word_counts.most_common(1)
        res_2.append(lbl.inverse_transform([yes[0][0]])[0])


    res = []
    for i in range(y_test_oofp.shape[0]):
        tmp = []
        for j in range(N):
            for clf in range(1):
                for n in range(clf_weigh[clf]):
                    tmp.append(int(y_test_oofp[i][j][clf]))
        res.append(max(set(tmp), key=tmp.count))

    result = pd.DataFrame()
    result['content_id'] = list(test_id)

    result['subject'] = list(res_2)
    result['subject'] = result['subject']

    result['sentiment_value'] = list(res)
    result['sentiment_value'] = result['sentiment_value'].astype(int)

    result['sentiment_word'] = ''
    result.to_csv('submit.csv', index=False)


def mul_subject():
    X, test, y, test_id, y1 = pre_process()
    N = 10
    kf = StratifiedKFold(n_splits=N, random_state=2018).split(X, y)

    # y_train_subject = np.zeros_like(y)
    # y_train_sentiment = np.zeros_like(y)

    y_test_oofp_sentiment = np.zeros((test.shape[0], N))
    y_test_oofp_subject = np.zeros((test.shape[0], N))

    y_test_subject = list()
    y_test_sentiment = list()
    for i in range(len(test_id)):
        y_test_subject.append(list())
        y_test_sentiment.append(list())

    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_val, sentiment_train, sentiment_val, subject_train, subject_val, = \
        X[train_fold, :], X[test_fold, :], y[train_fold], y[test_fold], y1[train_fold], y1[test_fold]
        clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
                                 max_depth=8, n_estimators=1, objective='binary',
                                 subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                                 learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
        clf.fit(X_train, sentiment_train)

        # sentiment_val_ = clf.predict(X_val)
        # y_train_oofp[test_fold] = sentiment_val
        # print('sentiment_value_f1:%f' % micro_avg_f1(label_validate, val_))
        # acc += micro_avg_f1(label_validate, val_)
        result = clf.predict(test)
        y_test_oofp_sentiment[:, i] = result

        # clf = svm.LinearSVC(loss='hinge', tol=1e-4, C=0.6)
        clf.fit(X_train, subject_train)
        # val_1 = clf.predict(X_validate)
        # y_train_oofp1[test_fold] = val_
        #
        # vcc += micro_avg_f1(label_1_validate, val_1)
        result = clf.predict(test)
        y_test_oofp_subject[:, i] = result
        print(0, i, 'finished')

    lbl = pk.load(open('../tmp/label_encoder.sav', 'rb'))
    for i in range(y_test_oofp_subject.shape[0]):
        tmp = []
        for j in range(N):
            tmp.append(int(y_test_oofp_subject[i][j]))
        word_counts = Counter(tmp)
        yes = word_counts.most_common(1)
        y_test_subject[i].append(lbl.inverse_transform([yes[0][0]])[0])

    for i in range(y_test_oofp_sentiment.shape[0]):
        tmp = []
        for j in range(N):
            tmp.append(y_test_oofp_sentiment[i][j])
        y_test_sentiment[i].append(max(set(tmp), key=tmp.count))

    kf = StratifiedKFold(n_splits=N, random_state=2018).split(X, y)
    for i in range(10):
        print(lbl.inverse_transform([i]))
        subject_oh = y1.copy()
        for l in range(len(subject_oh)):
            if subject_oh[l] != i:
                subject_oh[l] = 0
            else:
                subject_oh[l] = 1
        clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
                                 max_depth=8, n_estimators=100, objective='binary',
                                 subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                                 learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)

        clf.fit(X, subject_oh)
        result = clf.predict(test)
        for l in range(len(test_id)):
            if result[l] == 1 and lbl.inverse_transform([i]) not in y_test_subject[l]:
                y_test_subject[l].append(lbl.inverse_transform([i])[0])
                y_test_sentiment[l].append(0)
        print(1, i, 'finished')

    test_id_mul = list()
    subject_mul = list()
    sentiment_mul = list()
    for i in range(len(test_id)):
        for l in range(len(y_test_subject[i])):
            test_id_mul.append(test_id[i])
            subject_mul.append(y_test_subject[i][l])
            sentiment_mul.append(y_test_sentiment[i][l])

    result = pd.DataFrame()
    result['content_id'] = test_id_mul


    result['subject'] = subject_mul
    result['subject'] = result['subject']

    result['sentiment_value'] = sentiment_mul
    result['sentiment_value'] = result['sentiment_value'].astype(int)

    result['sentiment_word'] = ''
    result.to_csv('../submit.csv', index=False)

def cv_test_mul():
    train_vec = pd.read_csv('/home/hujoe/PycharmProjects/df-2018-NLP/content_vec_withoutD.csv', header=None)
    test_file = pd.read_csv('/home/hujoe/PycharmProjects/df-2018-NLP/data/test_public.csv')

    # train_vec_sentiment = pd.read_csv('../content_vec_sentiment.csv', header=None)
    train_vec = np.array(train_vec)
    # train_vec_sentiment = np.array(train_vec_sentiment)
    data = pd.read_csv('/home/hujoe/PycharmProjects/df-2018-NLP/data/train.csv')
    subject_vocab = list(['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观'])

    subject_list = list()
    for i in data['subject']:
        for k in range(10):
            if subject_vocab[k] == i:
                subject_list.append(k)
                break
    subject_list = np.array(subject_list)

    value_list = list()
    for i in data['sentiment_value']:
        value_list.append(i)
    value_list = np.array(value_list)

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
    test_id = list(test_file['content_id'])
    X, test, y, test_id, y1 = train_vec, test_vec, value_list, test_id, subject_list
    N = 10
    res = open('res2.txt', 'w')
    # kf = StratifiedKFold(n_splits=N, random_state=2018).split(X, y)
    for i in range(10):
        subject_oh = y1.copy()
        for l in range(len(subject_oh)):
            if subject_oh[l] != i:
                subject_oh[l] = 0
            else:
                subject_oh[l] = 1
        params = {'boosting_type': 'gbdt', 'num_leaves': 55, 'reg_alpha': 0.1, 'reg_lambda': 1,
                  'max_depth': 15, 'objective': 'binary',
                  'subsample': 0.8, 'colsample_bytree': 0.8, 'subsample_freq': 1,
                  'learning_rate': 0.06, 'min_child_weight': 1, 'random_state': 20, 'n_jobs': 4}

        data_train = lgb.Dataset(X, subject_oh)
        clf = lgb.cv(
            params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
            early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
        res.write(str(len(clf['rmse-mean'])))
        res.write(' ')
        res.write(str(clf['rmse-mean'][-1]))
        res.write('\n')

if __name__ == '__main__':
    run_base()
    # cv_test()