import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import jieba

if __name__ == '__main__':
    train_set = pd.read_csv('../data/train.csv')
    sentiment_words = train_set['sentiment_word']
    sentiment_values = train_set['sentiment_value']
    sentiment_words.fillna(0, inplace=True)
    stat = dict()
    stat['0'] = 0
    stat['-1'] = 0
    stat['1'] = 0
    for i in range(len(sentiment_words)):
        print(sentiment_words[i])
        stat[str(sentiment_values[i])] += 1
    print(stat)

