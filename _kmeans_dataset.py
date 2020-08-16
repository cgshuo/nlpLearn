#module_name, package_name, ClassName, method_name, ExceptionName,
#function_name, GLOBAL_VAR_NAME, instance_var_name,
# function_parameter_name, local_var_name.

import json
import math
import nltk
import numpy as np
import pandas as pd
import re
from numpy import mat
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
import time

# 发现问题， kmeans 与线性回归都是最小二乘法思想，当k为2时线性回归是kmeans的特例：kmeans无监督学习，线性回归有监督学习
#优化思路：时间主要耗费在了词矩阵构建上，可以预先处理存储，在kmeans中直接调用，会节省时间

def read_data(path): # 读dict格式的json数据
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            lines = json.loads(line) #loads： 是将string转换为dict
                                     #load： 是将里json格式字符串转化为dict，读取文件
                                     #dump： 是将dict类型转换为json格式字符串，存入文件
                                     #dumps： 是将dict转换为string
            data.append(lines)
    data = pd.DataFrame(data)
    return data

def save_nparray(numpy_array):
    np.save('file/kmeans_dataSet.npy', numpy_array, allow_pickle=True)

def text_tokenize(text): #分词
    text = re.sub(r'[^a-zA-Z0-9\s]', '', string=text)
    tokens = cut_model.tokenize(text)
    text = [token for token in tokens if token not in stopwords]
    return text

def word_in_text(corpus_df):
    wordsEmbedding = {}
    i=0
    for text in corpus_df:
        words = text_tokenize(text)
        for word in words:
            if word not in wordsEmbedding:
                i=i+1
                wordsEmbedding[word] = i
    return wordsEmbedding

def word_frequency(words): #统计词频
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1    # dict.get(key, default=None)如果value不存在返回default
    return counts

def cal_tfidf(corpus, wordsFrequency): #计算tfidf corpus:list;
    tf={} # 变量赋值是引用赋值
    tf.update(wordsFrequency)
    idf = {}
    tf_idf={}
    max_count = len(wordsFrequency)
    for key in tf.keys():
        tf[key] = tf[key]/max_count
    corpus_len = len(corpus)
    for text in corpus:
        for word in tf.keys():
            if word in text:
                idf[word] = idf.get(word, 0) + 1
    for key, value in idf.items():
        idf[key] = math.log(corpus_len/(value+1))
        tf_idf[key] = tf[key] * idf[key]
    return tf_idf

def extract_feature(tf_idf, count, wordsEmbedding): # 利用tf-idf值提取count个特征值及词频
    tfidf_list = sorted(tf_idf.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    tfidf_df = pd.DataFrame(tfidf_list[0:count])
    tfidf_df.columns = ['word', 'TFIDF']
    wordEmbedding=[]
    for word in tfidf_df['word']:
        wordEmbedding.append(wordsEmbedding[word])
    tfidf_df['wordsEmbedding'] = wordEmbedding
    return tfidf_df

def kmeans_cluster(dataSet, k): # dataSet为list
    sampleNum, col = dataSet.shape #dataSet的n*m n为记录数，m为纬度
    cluster = mat(np.zeros((sampleNum, 2))) #簇标记，记录最短就离
    centroids = np.zeros((k, col)) #k个圆心{C1...Ck}
    for i in range(k):
        index = int(np.random.uniform(0, sampleNum)) # 随即选择k个样本
        centroids[i, :] = dataSet[index, :]
    clusterChange = True # flag,当为False时，聚类结束
    while clusterChange:
        clusterChange = False
        for i in range(sampleNum):
            minDist = math.sqrt(sum(np.power(centroids[0, :] - dataSet[i, :], 2)))
            minIndex = 0
            #计算样本xi与均值向量cj的距离
            for j in range(1, k):
                distance = math.sqrt(sum(np.power(centroids[j, :] - dataSet[i, :],2)))
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 距离最近的均值向量cj确定xi的簇标记
            if cluster[i, 0] != minIndex:  # 属于minIndex簇(使distance最小)
                clusterChange = True
                cluster[i, :] = minIndex, minDist ** 2
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(cluster[:, 0].A == j)[0]] #dataSet[index],index=cluster中为cj的簇的点,[0]即为index
            if len(pointsInCluster) != 0:
                    centroids[j, :] = np.mean(pointsInCluster, axis=0) #求均值向量（用于计算距离聚类），axis不设置值，对m*n个数求均值，返回一个实数，axis = 0：压缩行，对各列求均值，返回 1* n 矩阵,axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
    return centroids,cluster

def plot_result(centroids, cluster, dataSet, k):
    sampleNum, col = dataSet.shape
    mark=['b', 'g', 'r', 'm', 'y', 'k', 'w']
    for i in range(sampleNum): #样本点聚类图
        markIndex = int(cluster[i, 0])
        plt.title('cluster')
        #plt.scatter(dataSet[i, 0], dataSet[i, 1], c=mark[markIndex], s=20, marker='o') #scatter画散点图，plot()画线性图
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    for i in range(k): #圆心
        plt.title('centroids')
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()

def NMI(cluser,):
    pass


# 获取特征值的df。然后转置，一行是一个数据样本，进行聚类（按行追加构建数据集）
if __name__ == '__main__':
    corpus_df = read_data("file/twitter_big_long.")
    stopwords = nltk.corpus.stopwords.words('english')
    cut_model = nltk.WordPunctTokenizer()

    allWords = word_in_text(corpus_df['text'])
    print('text counts:%d'%len(corpus_df) + ', word counts:%d'%len(allWords))

    count = 15  # 提取15个特征值
    feature_all_df=[]
    feature_all_df = pd.DataFrame(feature_all_df)
    for text in corpus_df['text']:
        words = text_tokenize(text)
        wordFrequency = word_frequency(words)
        tf_idf = cal_tfidf(np.array(corpus_df['text']).tolist(), wordFrequency)
        feature_df = extract_feature(tf_idf, count, allWords) #feature_df格式：word TFIDF wordsEmbedding
        featureValue_df = pd.DataFrame(feature_df[['word','TFIDF']].set_index('word')).T
        feature_all_df = pd.concat([feature_all_df, featureValue_df], axis=0, ignore_index=True, sort=False)
    dataSet = feature_all_df.fillna(0)
    dataSet = np.array(dataSet)
    save_nparray(dataSet)

    # # kmeans_cluster
    # k=6
    # start = time.clock()
    # centroids, cluster = kmeans_cluster(dataSet, k) #均值向量 与簇:xi=cj
    # sum=[0]*k
    # print("--------------------------------------")
    # print("手写kmeans结果")
    # for i in range(k):
    #     sum[i] = np.sum(np.array(cluster[:,0]) == i)
    #     print('第%d'%i + '类文本个数：%d'%sum[i])
    # end = time.clock()
    # print('用时%s' % end - start)
    # print("--------------------------------------")
    #
    # #sklearn库对比
    # start = time.clock()
    # kmeans = KMeans(n_clusters=k).fit(dataSet)
    # print("sklearn结果")
    # for i in range(k):
    #     sum[i] = np.sum(np.array(kmeans.labels_) == i)
    #     print('第%d'%i + '类文本个数：%d'%sum[i])
    # end = time.clock()
    # print('用时%s' % end - start)
    # print("--------------------------------------")
    #
    # #NMI 应该归一化，再计算，即最多的类都为1，而不是一个是1，一个是2
    # result_NMI = sklearn.metrics.normalized_mutual_info_score(np.array(cluster[:,0]), np.array(kmeans.labels_),average_method='arithmetic')
    # print("result_NMI:", 1-result_NMI)