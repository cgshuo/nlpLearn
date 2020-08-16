import pandas as pd
import numpy as np
import cntopic
import os
import jieba

def Latent_Dirichlet_Allocation():
    pass


if __name__ == '__main__':
    # cntopic_model_LDA
    corpus_content = np.load('file/LDA_corpus_content_tokens.npy', allow_pickle=True)
    documents =corpus_content.tolist()
    topic = cntopic.Topic(cwd=os.getcwd())  # 构建词典dictionary
    topic.create_dictionary(documents=documents)  # 根据documents数据，构建词典空间
    topic.create_corpus(documents=documents)  # 构建语料(将文本转为文档-词频矩阵)
    topic.train_lda_model(n_topics=10)  # 指定n_topics，构建LDA话题模型

    document = jieba.lcut('游戏体育真有意思')
    topic.get_document_topics(document)
