import pandas as pd

def diff():
    baseline = pd.read_csv('~/baseline.csv')
    test = pd.read_csv('submit_1.csv')
    train_set = pd.read_csv('../data/test_public.csv')
    for i in range(baseline.shape[0]):
        if baseline['subject'][i] != test['subject'][i] or baseline['sentiment_value'][i] != test['sentiment_value'][i]:
            print(baseline['content_id'][i], baseline['subject'][i], baseline['sentiment_value'][i], test['subject'][i], test['sentiment_value'][i], train_set['content'][i])

def diff_mul():
    baseline = pd.read_csv('../res_set/res0.6.csv')
    test = pd.read_csv('../res_set/res0.8.csv')
    train_set = pd.read_csv('../data/test_public.csv')
    j = 0
    sum = 0
    for i in range(len(test['content_id'])):
        while test['content_id'][i] != baseline['content_id'][j]:
            sum += 1
            print(baseline['content_id'][j], baseline['subject'][j], baseline['sentiment_value'][j])
            for k in range(len(train_set['content_id'])):
                if train_set['content_id'][k] == baseline['content_id'][j]:
                    print(train_set['content'][k])
                    break
            j += 1
        j += 1
    print(sum)

def diff_sent():
    baseline = pd.read_csv('../res.csv')
    test = pd.read_csv('../data/test_public.csv')
    j = 0
    for i in range(len(test['content_id'])):
        t = 0
        s = 0
        while baseline['content_id'][j] != test['content_id'][i]:
            if t == 0:
                if baseline['sentiment_value'][j-1] == 0:
                    s = 1
                else:
                    print(baseline['subject'][j-1], ' ', baseline['sentiment_value'][j-1])
                t = 1
            if not s:
                print(baseline['content_id'][j], ' ', baseline['subject'][j], ' ', baseline['sentiment_value'][j], ' ', test['content'][i-1])
            j += 1
        j += 1

def stat():
    baseline = pd.read_csv('submit_1.csv')
    zeros = 0
    minus = 0
    ones = 0
    sentiment = baseline['sentiment_value']
    for i in sentiment:
        if i == 1:
            ones += 1
        elif i == -1:
            minus += 1
        else:
            zeros += 1
    sum = minus + zeros + ones
    print(minus/sum, zeros/sum, ones/sum)

if __name__ == '__main__':
    diff_mul()