import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import jieba
from matplotlib import pyplot as plt
def sentiment():
    train_set = pd.read_csv('../data/train.csv')
    sentiment_value = train_set['sentiment_value']
    subject = train_set['subject']
    test_id = train_set['content_id']
    str = ''
    l = 0
    h_sentiment_value = -1
    h_subject = -1
    for i in range(len(test_id)):
        if test_id[i] == str:
            if (l == 0):
                print('\n')
                print(str, h_subject, h_sentiment_value)
                l = 1
            print(str, subject[i], sentiment_value[i])
        else:
            l = 0
        h_subject = subject[i]
        h_sentiment_value = sentiment_value[i]
        str = test_id[i]

def mul():
    train_set = pd.read_csv('../data/train.csv')
    content_id = train_set['content_id']
    content = train_set['content']
    sentences = []
    leng = []
    for i in content:
        # sentences.append(list(jieba.cut(i)))
        # leng.append(list(jieba.cut(i)).__len__())
        leng.append(len(i))
    last = ''
    sum = 0
    l = []
    i = 0
    while i < len(content_id) - 1:
        k = False
        while content_id[i] == last and i < 9946:
            sum += 1
            i += 1
            print(content_id[i-1])
            if not k:
                l.append(leng[i-1])
                k = True
        last = content_id[i]
        i += 1
    print(sum)
    x = np.array(l)
    k = 0
    for i in l:
        if i < 25:
            k += 1
    print(k)

def test():
    train_set = pd.read_csv('../res_set/res0.6.csv')
    test = pd.read_csv('../data/test_public.csv')
    content_id = train_set['content_id']
    content = test['content']
    test_id = test['content_id']
    leng = []
    for i in content:
        leng.append(len(i))
    last = ''
    sum = 0
    l = []
    i = 0
    while i < len(content_id) - 1:
        k = False
        while content_id[i] == last and i < 9946:
            sum += 1
            i += 1
            print(content_id[i-1])
            if not k:
                for ii in range(len(test_id)):
                    if test_id[ii] == content_id[i-1]:
                        l.append(leng[ii])
                        break
                k = True
        last = content_id[i]
        i += 1
    print(sum)
    x = np.array(l)
    k = 0
    for i in l:
        if i < 25:
            k += 1
    print(k)

if __name__ == '__main__':
    test()