#coding=utf-8
#Author:Cgshuo
#Date:2020-8-20
import numpy as np

#记录一个很蠢的bug:在corpus最后加了回车，用ord()计算ascll码时报错 IndexError: string index out of range

def loadArticle(fileName):

    #初始化文章列表
    artical = []
    #打开文件
    fr = open(fileName, encoding='utf-8')
    #按行读取文件
    for line in fr.readlines():
        #读到的每行最后都有一个\n，使用strip将最后的回车符去掉
        line = line.strip()
        #将该行放入文章列表中
        artical.append(line)
    return artical

def trainModel(fileName):
    # return HMM = （A, B, PI）
    # 定义4中状态。B：词语开头 M：词语中间 E：词语结尾 S：非词语，单字
    statuDict = {'B':0, 'M':1, 'E':2, 'S':3}
    #初始化状态转移矩阵A 4*4
    A = np.zeros((4, 4))
    #初始化观测概率矩阵B，4种状态，65536个汉字上限 4*65536
    B = np.zeros((4, 65536))
    #4种状态先验概率矩阵PI 1*4
    PI = np.zeros(4)

    #统计次数
    file = open(fileName, encoding='utf-8')
    for line in file.readlines():
        curLine = line.strip().split()
        # lineWord = line.replace(" ","")
        wordLabel = []
        for i in range(len(curLine)):
            if len(curLine[i]) == 1:
                #如果词长度为1是单个字，标记为S
                label = 'S'
            else:
                #如果长度不为1，开头为B，结尾为E，中间len-2个M
                label = 'B' + 'M' * (len(curLine[i]) - 2) + 'E'

            #如果是单行开头第一个字，PI对应位置+1
            if i == 0: PI[statuDict[label[0]]] += 1

            for j in range(len(label)):
                #遍历状态链中每一个状态，并在B找到对应的中文汉字，ord()返回ascll数值
                B[statuDict[label[j]]][ord(curLine[i][j])] += 1
            wordLabel.extend(label)


        for i in range(1, len(wordLabel)):
            #统计t时刻 t+1时刻的状态组合出现次数
            A[statuDict[wordLabel[i - 1]]][statuDict[wordLabel[i]]] += 1

    #计算频率，最大相似估计化为概率
    #计算A
    for i in range(len(A)):
        sum = np.sum(A[i])
        for j in range(len(A[i])):
            # log0没有意义，手动赋极小值
            if A[i][j] == 0 :
                A[i][j] = -3.14e+100
            else:
                A[i][j] = np.log(A[i][j]/sum)
    #计算B
    for i in range(len(B)):
        sum = np.sum(B[i])
        for j in range(len(B[i])):
            # log0没有意义，手动赋极小值
            if B[i][j] == 0 :
                B[i][j] = -3.14e+100
            else:
                B[i][j] = np.log(B[i][j]/sum)
    #计算PI
    sum = np.sum(PI)
    for i in range(len(PI)):
        if PI[i] == 0:
            PI[i] = -3.14e+100
        else:
            PI[i] = np.log(PI[i] / sum)
    return A, B, PI

def participle(article, A, B, PI):
    # HMM=(A,B,PI);算法：Viterbi Algorithm

    retArtical=[]
    for line in article:
        #初始化δ 状态S1...S4, 观测O=(O1..OT) δ：T*4矩阵 (4个维度=4个状态）
        delta = [[0 for i in range(4)] for i in range(len(line))]
        for i in range(4):
            #初始化状态链起点的四个状态的概率;
            #注：在计算A、B、PI时已经完成了log运算，所以公式的乘δ=PI*B 都变成了logδ=logPI+logB了
            delta[0][i] = PI[i] + B[i][ord(line[0])]
        #初始化ψ，初始时为0
        psi = [[0 for i in range(4)] for i in range(len(line))]

        #递推
        for t in range(1, len(line)):
            for i in range(4):
                #初始化一个存储临时的状态概率，取出其中的最大值
                #临时状态及索引：'B': 0, 'M': 1, 'E': 2, 'S': 3
                tmpDelta = [0] * 4
                for j in range(4):
                    tmpDelta[j] = delta[t-1][j] + A[j][i]
                maxDelta = max(tmpDelta)
                #记录索引以便回溯
                maxDeltaIndex = tmpDelta.index(maxDelta)

                delta[t][i] = maxDelta + B[i][ord(line[t])]
                psi[t][i] = maxDeltaIndex

        #状态链
        sequence = []
        #最大概率的索引i
        i_opt = delta[len(line) - 1].index(max(delta[len(line) - 1]))
        sequence.append(i_opt)
        #回溯
        for t in range(len(line) - 1, 0, -1):
            i_opt = psi[t][i_opt]
            sequence.append(i_opt)
        sequence.reverse()

        #分词
        curLine = ''
        for i in range(len(line)):
            curLine += line[i]
            # 如果该字是3：S->单个词  或  2:E->结尾词 ，则在该字后面加上分隔符 |
            # 此外如果改行的最后一个字了，也就不需要加 |
            if (sequence[i] == 3 or sequence[i] == 2) and i != (len(line) - 1):
                curLine += '|'
        retArtical.append(curLine)
    return retArtical

if __name__ == '__main__':
    #trian
    A, B, PI = trainModel('file/pku_training.utf8')

    artical = loadArticle('file/pku_test.utf8')

    #test
    partiArtical = participle(artical, A, B, PI)

    print('-------------------分词结果----------------------')
    for line in partiArtical:
        print(line)




