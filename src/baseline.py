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
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pickle as pk
from sklearn import svm


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
    return ' '.join(word)


def pre_process():
    data, nrw_train, y, test_id, y1 = get_data()

    data['cut_comment'] = data['content'].map(processing_data)

    print('TfidfVectorizer')
    tf = TfidfVectorizer(ngram_range=(1, 2), analyzer='char')
    discuss_tf = tf.fit_transform(data['cut_comment'])

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
    params = { 'boosting_type':'gbdt', 'num_leaves':80, 'reg_alpha':0.1, 'reg_lambda':1,
              'max_depth':8, 'objective':'binary',
              'subsample':0.8, 'colsample_bytree':0.8, 'subsample_freq':1,
              'learning_rate':0.06, 'min_child_weight':1, 'random_state':20, 'n_jobs':4}

    X, test, y, test_id, y1 = pre_process()
    data_train = lgb.Dataset(X, y)
    clf = lgb.cv(
        params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
        early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

    print(str(len(clf['rmse-mean'])))

    data_train = lgb.Dataset(X, y1)

    clf = lgb.cv(
        params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
        early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

    print(str(len(clf['rmse-mean'])))

def run_base():
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
        X_train, X_validate, label_train, label_validate, label_1_train, label_1_validate, = X[train_fold, :], X[
                                                                                                               test_fold,
                                                                                                               :], y[
                                                                                                 train_fold], y[
                                                                                                 test_fold], y1[
                                                                                                 train_fold], y1[
                                                                                                 test_fold]
        clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
                                 max_depth=8, n_estimators=500, objective='binary',
                                 subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                                 learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
        clf.fit(X_train, label_train)

        val_ = clf.predict(X_validate)
        y_train_oofp[test_fold] = val_
        print('sentiment_value_f1:%f' % micro_avg_f1(label_validate, val_))
        acc += micro_avg_f1(label_validate, val_)
        result = clf.predict(test)
        y_test_oofp[:, i] = result

        clf = svm.LinearSVC(loss='hinge', tol=1e-4, C=0.6)
        clf.fit(X_train, label_1_train)
        val_1 = clf.predict(X_validate)
        y_train_oofp1[test_fold] = val_

        vcc += micro_avg_f1(label_1_validate, val_1)
        result = clf.predict(test)
        y_test_oofp_1[:, i] = result

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
                                 max_depth=8, n_estimators=500, objective='binary',
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

    lbl = pk.load(open('label_encoder.sav', 'rb'))
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
    result.to_csv('/home/hujoe/PycharmProjects/df-2018-NLP/submit.csv', index=False)

if __name__ == '__main__':
    mul_subject()