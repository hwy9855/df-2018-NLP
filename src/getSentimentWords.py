import pandas as pd

if __name__ == '__main__':
    train_set = pd.read_csv('../data/train.csv')
    sentiment_words = train_set['sentiment_word']
    tmp = list()
    for word in sentiment_words:
        if word not in tmp:
            tmp.append(word)

    txt = open('../data/sentiment_words.txt', 'w')
    for w in tmp:
        txt.write(str(w))
        txt.write('\n')