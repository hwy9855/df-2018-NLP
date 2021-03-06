from lightgbm import LGBMClassifier
from sklearn import svm
import numpy as np
from sklearn.model_selection import StratifiedKFold

class Lgb():

    @staticmethod
    def cal_subject(train_vec, train_subject, test_id, test_vec, iter):
        # param = { 'boosting_type':'gbdt', 'num_leaves':55, 'reg_alpha':0.0, 'reg_lambda':1,
        #           'max_depth':15, 'n_estimators':6000, 'objective':'binary',
        #           'subsample':0.8, 'colsample_bytree':0.8, 'subsample_freq':1,
        #           'learning_rate':0.06, 'min_child_weight':1, 'random_state':20, 'n_jobs':4}
        # clf = LGBMClassifier(param)
        # clf = svm.LinearSVC(max_iter=100000)
        clf = LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
                             max_depth=8, n_estimators=iter, objective='binary',
                             subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                             learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
        # iter_list = [803,61,69,314,196,223,64,153,55,284,173]

        test_res = list()
        for l in range(len(test_id)):
            test_res.append(list())
        for i in range(10):
            train_label_onehot = train_subject.copy()
            for l in range(len(train_subject)):
                if train_subject[l] != i:
                    train_label_onehot[l] = 0
                else:
                    train_label_onehot[l] = 1
            # print(train_label_onehot)
            # print(train_subject)
            # clf = LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
            #                      max_depth=8, n_estimators=iter_list[i], objective='binary',
            #                      subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
            #                      learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
            clf.fit(train_vec, train_label_onehot)
            res_onehot = clf.predict(test_vec)
            for l in range(len(test_id)):
                if res_onehot[l] == 1:
                    test_res[l].append(i)
        # clf = LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
        #                      max_depth=8, n_estimators=iter_list[10], objective='binary',
        #                      subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        #                      learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
        clf.fit(train_vec, train_subject)
        res = clf.predict(test_vec)
        for l in range(len(test_id)):
            if res[l] not in test_res[l]:
                test_res[l].append(res[l])

        return test_id, test_res

    @staticmethod
    def cal_subject_mul(train_vec, train_subject, test_id, test_vec, iter, baseline):
        # param = { 'boosting_type':'gbdt', 'num_leaves':55, 'reg_alpha':0.0, 'reg_lambda':1,
        #           'max_depth':15, 'n_estimators':6000, 'objective':'binary',
        #           'subsample':0.8, 'colsample_bytree':0.8, 'subsample_freq':1,
        #           'learning_rate':0.06, 'min_child_weight':1, 'random_state':20, 'n_jobs':4}
        # clf = LGBMClassifier(param)
        # clf = svm.LinearSVC(max_iter=100000)
        N = 10
        train_vec = np.array(train_vec)
        train_subject = np.array(train_subject)
        kf = StratifiedKFold(n_splits=N, random_state=2018).split(train_vec, train_subject)
        clf = LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
                             max_depth=8, n_estimators=iter, objective='binary',
                             subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                             learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
        # iter_list = [803,61,69,314,196,223,64,153,55,284,173]

        test_res = list()

        for l in range(len(test_id)):
            test_res.append(list())

        subject_vocab = {'价格':0, '配置':1, '操控':2, '舒适性':3, '油耗':4,
                         '动力':5, '内饰':6, '安全性':7, '空间':8, '外观':9}

        for l in range(len(test_id)):
            test_res[l].append(subject_vocab[baseline['subject'][l]])

        value_list = list()

        for l in range(len(test_id)):
            value_list.append(list())

        for l in range(len(test_id)):
            value_list[l].append(baseline['sentiment_value'][l])

        res_sub = np.zeros([10, len(test_id), N])
        for k, (train_fold, test_fold) in enumerate(kf):
            for i in range(10):
                train_subject_kf = train_subject[train_fold]
                train_label_onehot = train_subject_kf.copy()
                for l in range(len(train_subject_kf)):
                    if train_subject_kf[l] != i:
                        train_label_onehot[l] = 0
                    else:
                        train_label_onehot[l] = 1
                # print(train_label_onehot)
                # print(train_subject)
                # clf = LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
                #                      max_depth=8, n_estimators=iter_list[i], objective='binary',
                #                      subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                #                      learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
                clf.fit(train_vec[train_fold], train_label_onehot)
                res_onehot = clf.predict(test_vec)
                for l in range(len(test_id)):
                    res_sub[i][l][k] = res_onehot[l]

        res_onehot = np.zeros([10, len(test_id)])
        for i in range(10):
            for j in range(len(test_id)):
                tmp = []
                for k in range(N):
                    tmp.append(res_sub[i][j][k])
                if sum(tmp) > 7:
                    res_onehot[i][j] = 1

        for i in range(10):
            for l in range(len(test_id)):
                if res_onehot[i][l] == 1 and i not in test_res[l]:
                    test_res[l].append(i)
                    value_list[l].append(0)
                    # value_list[l].append(value_list[l][0])
        # clf = LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
        #                      max_depth=8, n_estimators=iter_list[10], objective='binary',
        #                      subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        #                      learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)



        return test_id, test_res, value_list

