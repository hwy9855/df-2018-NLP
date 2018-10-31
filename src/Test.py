import pandas as pd

if __name__ == '__main__':
    final = pd.read_csv('../res_set/res_final.csv')
    s = dict()
    v = dict()
    ID = dict()
    for i in range(final['content_id'].shape[0]):
        if final['content_id'][i] not in ID:
            ID[final['content_id'][i]] = 1
        else:
            ID[final['content_id'][i]] += 1
        if final['subject'][i] not in s:
            s[final['subject'][i]] = 1
        else:
            s[final['subject'][i]] += 1
        if final['sentiment_value'][i] not in v:
            v[final['sentiment_value'][i]] = 1
        else:
            v[final['sentiment_value'][i]] += 1
    print(s, v, len(ID))
