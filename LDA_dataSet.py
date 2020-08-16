#LDA 三个步骤：生成话题的单词分布、生成文本的话题分布、生成文本的单词序列

import jieba
import pandas as pd
import numpy as np

def text2tokens(corpus):
    tokens = jieba.lcut(corpus)
    tokens = [word for word in tokens if len(word) > 1 and word not in stopwords]
    return tokens

def save_data(nparray):
    np.save('file/LDA_corpus_content_tokens.npy', nparray, allow_pickle=True)

if __name__ == '__main__':
    corpus_df = pd.read_csv('file/lda_chinese_news.csv')
    stopwords = [line.strip() for line in open('file/stopwords.txt', 'r', encoding='utf-8').readlines()]
    corpus_content = [text2tokens(content) for content in corpus_df['content']]
    save_data(np.array(corpus_content))