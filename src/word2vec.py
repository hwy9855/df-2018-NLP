from gensim.models import word2vec
import re
import jieba
import pandas as pd


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


if __name__ == '__main__':
    jieba.load_userdict('../data/dict.txt')
    data = pd.read_csv('../data/train.csv')
    contents = data['content']
    # stop_txt = open('../data/stop_words.txt', 'r')
    # stop_lines = stop_txt.readlines()
    # stop_set = list()
    # for line in stop_lines:
    #     stop_set.append(line.split('\n')[0])
    content_sentences = list()
    for sentence in contents:
        cut_sentence = jieba.cut(sentence)
        # dele = list()
        # for word in cut_sentence:
        #     if str(word) in stop_set:
        #         dele.append(word)
        #     elif hasNumbers(word):
        #         dele.append(word)
        # for word in dele:
        #     cut_sentence.remove(word)
        content_sentences.append(' '.join(cut_sentence))

    with open('../content.txt', 'w') as f:
        for sentence in content_sentences:
            for word in sentence:
                f.write(word)
                f.write(' ')
            f.write('\n')

    content_sentences = word2vec.Text8Corpus('../content.txt')
    model = word2vec.Word2Vec(content_sentences, size=200)
    # test = model.most_similar('油耗', topn=20)
    print(model.wv.vocab)