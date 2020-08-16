import os
import jieba
import math
import numpy as np

#计算文本相似度：计算词频，计算TF-IDF，降序排列选前20个词建立词向量
#将每一个txt存为一个str，抽取txt的词向量，然后比较txt之间的相似度

def get_filelist(path,Filelist):
    if os.path.isfile(path):
        Filelist.append(path)
    elif os.path.isdir(path):
        for s in os.listdir(path):
            new_path = os.path.join(path,s)
            get_filelist(new_path,Filelist)
    return Filelist

def get_tfidf(data,Filelist): #通过TFIDF计算一个特征的权重时，该权重体现出的根本不是特征的重要程度！不会拿TFIDF去做特征选择
    tf_idf={}
    idf = {}
    tf = {}

    #tf
    seg_list = jieba.cut(data, cut_all=False)  # 精确模式
    for word in seg_list:
        if word not in stopwords:
            if not word in tf:
                tf[word] = tf.get(word, 1)
                idf[word] = idf.get(word, 0)    # 初始化idf
            tf[word] = tf.get(word)+1
    word_max = max(tf.values())
    for key in tf.keys():
        tf[key]= tf[key]/word_max  # tf = 该词文章出现次数/文章出现最多的词个数

    #idf
    count = len(Filelist)
    for path in Filelist:
        with open(path, 'r', encoding='gb18030') as f:
            text = f.read()
            for key in tf.keys:
                if key in text:
                    idf[key] += 1
    for key,value in idf.items():
        idf[key] = math.log(count/(value+1))  # idf = log(语料库文档总数/(包含该词的文档数+1）
    for key,value in tf_idf.items():
        tf_idf[key] = tf[key] * idf[key]    # tf_idf = tf(x) * idf(x)
    tfidf_list = sorted(tf_idf.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    tfidf_list = tfidf_list[0:20]
    return tfidf_list

def cal_simliarity(x,y):
    num = x.dot(y.T) #x与y的转置矩阵相乘
    denom = np.linalg.norm(x) * np.linalg.norm(y) #numpy.linalg模块包含线性代数的函数,.norm代表范数（模）
    return num / denom


if __name__ =='__main__':

    #Read
    text = ""
    corpus = [] # 语料库
    path = "/Users/caoguangshuo/Downloads/第一阶段要求和相关的数据/dataset"
    stopwords = [line.strip() for line in open('file/stopwords.txt', 'r', encoding='utf-8').readlines()]
    File_list = get_filelist(path,[])
    for e in File_list:
        with open(e,'r',encoding='gb18030') as f: #gb18030
            data = f.read()
            text+=data
            corpus.append(get_tfidf(data,File_list))
    print(corpus)

    #Tokenization and word frequency with dict
    text=text.replace(" ","")
    dicts={}    # dictionary
    word_fre={}     # word frequency
    all_seg_list = jieba.cut(text, cut_all=False) # 精确模式
    i=0
    for word in all_seg_list:
        if word not in stopwords:
            if not word in dicts:
                word_fre[word] = word_fre.get(word, 1)
                i=i+1
                dicts[word]=i   # dictionary for replace
            word_fre[word] = word_fre.get(word)+1
    word_fre=sorted(word_fre.items(), key=lambda a: a[1],reverse=True)  # words count of all news



    # #Replace 作业1
    # for e in File_list:
    #     with open(e,'r+',encoding='gb18030') as f: # r+追加 w+覆盖
    #         data = f.read()
    #         single_seg_list = jieba.cut(data, cut_all=False)
    #         for word in  single_seg_list:
    #             if word in dicts:
    #                 data=data.replace(word, str(dicts[word]) + " ") #replace出一个新的 不改变以前的
    #         f.write(data)

    #word_list = np.c_[np.array([i for i in dicts.keys()]),np.zeros(len(dicts))]


