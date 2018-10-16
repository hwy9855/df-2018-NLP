import pandas as pd


class GetResult():


    @staticmethod
    def __get_subject_vocab():
        subject_vocab = list(['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观'])
        return subject_vocab

    @staticmethod
    def __get_subject_list():
        subject_vocab = GetResult.__get_subject_vocab()
        data = pd.read_csv('../data/train.csv')
        subject_list = list()
        for i in data['subject']:
            for k in range(10):
                if subject_vocab[k] == i:
                    subject_list.append(k)
                    break
        return subject_list

    @staticmethod
    def cal_F1(res_id, res_subject, K):
        data = pd.read_csv('../data/train.csv')
        subject_list = GetResult.__get_subject_list()
        real_subjcet = subject_list[K:]
        real_id = list(data['content_id'])[K:]
        test_subject = list()
        for l in range(len(res_id)):
            test_subject.append(list())
        test_id = list()
        i = -1
        for l in range(len(real_id)):
            if real_id[l] not in test_id:
                test_id.append(real_id[l])
                i += 1
            test_subject[i].append(real_subjcet[l])

        err = 0
        corr = 0

        for l in range(len(res_subject)):
            for s in res_subject[l]:
                if s in test_subject[l]:
                    corr += 1
                else:
                    err += 1

        Tp = corr
        Fp = err
        Fn = 500-Tp-Fp

        P = Tp / (Tp + Fp)
        R = Tp / (Tp + Fn)

        F1 = 2 * P * R / (P + R)

        print(corr, err, F1)

    @staticmethod
    def res2doc(res_id, res_subject):
        subject_voacb = GetResult.__get_subject_vocab()
        res = open('../res.csv', 'w')
        res.write('content_id,subject,sentiment_value,sentiment_word\n')
        for i in range(len(res_id)):
            for s in res_subject[i]:
                res.write(res_id[i])
                res.write(',')
                res.write(subject_voacb[s])
                res.write(',')
                res.write('0,\n')

    @staticmethod
    def res2doc_mul(res_id, res_subject, value_list):
        subject_voacb = GetResult.__get_subject_vocab()
        res = open('../res.csv', 'w')
        res.write('content_id,subject,sentiment_value,sentiment_word\n')
        for i in range(len(res_id)):
            for l in range(len(res_subject[i])):
                res.write(res_id[i])
                res.write(',')
                res.write(subject_voacb[res_subject[i][l]])
                res.write(',')
                res.write(str(value_list[i][l]))
                res.write(',\n')

    @staticmethod
    def res2doc_sentiment(res_id, res_subject, res_sentiment):
        subject_voacb = GetResult.__get_subject_vocab()
        res = open('../res_sentiment.csv', 'w')
        res.write('content_id,subject,sentiment_value,sentiment_word\n')
        for i in range(len(res_id)):
            res.write(res_id[i])
            res.write(',')
            res.write(subject_voacb[res_subject[i]])
            res.write(',')
            res.write(str(int(res_sentiment[i])))
            res.write(',\n')

