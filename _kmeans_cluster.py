import numpy as np
import sklearn
from sklearn.cluster import KMeans
from _kmeans_dataset import kmeans_cluster

if __name__ == '__main__':
    dataSet = np.load('file/kmeans_dataSet.npy', allow_pickle=True)
    # kmeans_cluster
    k = 6
    centroids, cluster = kmeans_cluster(dataSet, k)  # 均值向量 与簇:xi=cj
    sum = [0] * k
    print("--------------------------------------")
    print("手写kmeans结果:")
    for i in range(k):
        sum[i] = np.sum(np.array(cluster[:, 0]) == i)
        print('第%d' % i + '类文本个数：%d' % sum[i])
    print("--------------------------------------")

    # sklearn库对比
    kmeans = KMeans(n_clusters=k).fit(dataSet)
    print("sklearn结果:")
    for i in range(k):
        sum[i] = np.sum(np.array(kmeans.labels_) == i)
        print('第%d' % i + '类文本个数：%d' % sum[i])
    #print('用时%s' % end - start)
    print("--------------------------------------")

    # NMI 应该归一化，再计算，即最多的类都为1，而不是一个是1，一个是2
    A=np.array(cluster[:, 0]).flatten().astype(int)
    B=np.array(kmeans.labels_)
    NMI = sklearn.metrics.normalized_mutual_info_score(B, A, average_method='arithmetic')
    print("NMI:", 1-NMI)
    print("--------------------------------------")