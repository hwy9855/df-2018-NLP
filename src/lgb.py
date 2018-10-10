from lightgbm import LGBMClassifier
from sklearn import svm
import numpy as np


class Lgb():

    @staticmethod
    def cal_subject(train_vec, train_subject, test_id, test_vec, iter):
        # param = { 'boosting_type':'gbdt', 'num_leaves':55, 'reg_alpha':0.0, 'reg_lambda':1,
        #           'max_depth':15, 'n_estimators':6000, 'objective':'binary',
        #           'subsample':0.8, 'colsample_bytree':0.8, 'subsample_freq':1,
        #           'learning_rate':0.06, 'min_child_weight':1, 'random_state':20, 'n_jobs':4}
        # clf = LGBMClassifier(param)
        # clf = svm.LinearSVC(max_iter=100000)
        # clf = LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
        #                      max_depth=8, n_estimators=iter, objective='binary',
        #                      subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        #                      learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
        iter_list = [243,68,91,73,109,70,46,80,51,83,173]

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
            clf = LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
                                 max_depth=8, n_estimators=iter_list[i], objective='binary',
                                 subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                                 learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
            clf.fit(train_vec, train_label_onehot)
            res_onehot = clf.predict(test_vec)
            for l in range(len(test_id)):
                if res_onehot[l] == 1:
                    test_res[l].append(i)
        clf = LGBMClassifier(boosting_type='gbdt', num_leaves=80, reg_alpha=0.1, reg_lambda=1,
                             max_depth=8, n_estimators=iter_list[10], objective='binary',
                             subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                             learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
        clf.fit(train_vec, train_subject)
        res = clf.predict(test_vec)
        for l in range(len(test_id)):
            if res[l] not in test_res[l]:
                test_res[l].append(res[l])

        return test_id, test_res

