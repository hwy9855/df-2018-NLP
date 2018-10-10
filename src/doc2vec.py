import pandas as pd
import numpy as np
import jieba


class Doc2Vec():

    @staticmethod
    def __get_idx(vocab, word):
        for i in range(len(vocab)):
            if vocab[i] == word:
                return i
        return -1

    @staticmethod
    def __doc2vec(docs, vocab):
        d = len(docs)
        w = len(vocab)
        content_vec = np.zeros([d, w])
        i = 0
        for doc in docs:
            for word in doc:
                if word in vocab:
                    content_vec[i][Doc2Vec.__get_idx(vocab, word)] += 1
            i += 1
        return content_vec

    @staticmethod
    def test2vec():
        # 添加关键词
        jieba.load_userdict('../data/dict.txt')
        # 读入数据
        data = pd.read_csv('../data/test_public.csv')
        vocab_txt = open('../vocab_withoutD.txt')
        vocab = list()
        lines = vocab_txt.readlines()
        for line in lines:
            vocab.append(line.split('\n')[0])
        content = data['content']
        content_list = list()
        for i in range(content.count()):
            content_list.append(list(jieba.cut(content[i])))
        test_vec = Doc2Vec.__doc2vec(content_list, vocab)
        return test_vec

    @staticmethod
    def test2vec_sentiment():
        # 添加关键词
        jieba.load_userdict('../data/sentiment_words.txt')
        # 读入数据
        data = pd.read_csv('../data/test_public.csv')
        vocab_txt = open('../vocab_sentiment.txt')
        vocab = list()
        lines = vocab_txt.readlines()
        for line in lines:
            vocab.append(line.split('\n')[0])
        content = data['content']
        content_list = list()
        for i in range(content.count()):
            content_list.append(list(jieba.cut(content[i])))
        test_vec = Doc2Vec.__doc2vec(content_list, vocab)
        return test_vec