# -*- coding:utf-8 -*-
# @Time: 2020/8/25 10:22 AM
# @Author: cgshuo
# @Email: cgshuo@163.com
# @File: BPNN.py

import numpy as np
import tensorflow
from cnews_loader import read_category, read_vocab, process_file
"""
    构建一个隐层的BPNN 
    输出为5个分类
"""

def initialize_parameters(x_n, h_n, y_n):
    """
    1。初始化参数
    :param x_n: x为输入矩阵的特征值个数
    :param h_n: h为隐藏层的大小
    :param y_n: y为可输出的分类个数
    :return:  parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    """
    np.random.seed(2) # 让两次随机的取值一样
    w1 = np.random.randn(h_n, x_n) * 0.01  # randn()使随机值服从高斯分布
    b1 = np.zeros((h_n, 1))
    w2 = np.random.randn(y_n, h_n) * 0.01 # y*h 一个y需要h个w2
    b2 = np.zeros(y_n, 1)

    #通过字典存储参数
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters

def forward_propagation(X, parameters):
    """
    2。利用网络和初始化参数，正向计算（z,a,y) z=w*a+b, a=ψ(z)
    :param X: 输入的特征矩阵
    :param parameters: 即随机初始化的(w,b)
    :return: a2, cache= {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    """
    w1 = parameters['w1'] # h*x (h隐层大小，x为特征维度）
    b1 = parameters['b1']
    w2 = parameters['w2'] # y*h 一个y需要h个w2
    b2 = parameters['b2']

    z1 = np.dot(w1, X) + b1
    a1 = np.tanh(z1)            #第一层的激活函数，使用tanh()
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))  #第二层的激活函数，使用sigmod()

    #通过字典存储参数
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    return a2, cache

def compute_cost(a2, Y, parameters):
    """
    3。计算损失函数
    :param a2: 3层神经网络的最后一层输出（y=a2=ψ(z2=w2*a1+b2))
    :param Y:  最后一层的输出矩阵，y_n*x_n 即行数为类别，列数为样本数
    :param parameters: 初始化的parameters（w,b)
    :return: cost 采用交叉熵cross-entropy作为代价函数
    """
    m = Y.shape[1] #样本数

    logprobs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cost = - np.sum(logprobs) / m

    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    4。逆传播算法（BP）步骤：(1)步骤1。初始化(w,b) (2)步骤2。前向传播求(z, a, y) (3)链式法求偏导数
    梯度下降求局部极值，沿着梯度的方向寻找，梯度的下降速度为 wk+1=wk - å*df(w)/dw
    :param parameters: 初始化的(w,b)
    :param cache: 经过前向计算得出的（z, a, y)
    :param X: 样本X
    :param Y: 类别Y
    :return: 对w和b的偏导数
    """
    m = Y.shape[1]
    w2 = parameters['w2']
    a1 = cache['a1']
    a2 = cache['a2']

    #逆传播计算dw1, db1, dw2, db2
    dz2 = a2 - Y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    return grads

def update_parameters(parameters, grads, learning_rate=0.4):
    """
    5。更新参数，梯度下降求局部极值
    :param parameters: (w,b)
    :param grads: 偏导数{'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}
    :param learning_rate: 步长参数å（学习率>0)
    :return: 迭代更新parameters{'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    """
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    # 更新参数
    w1 = w1 - dw1 * learning_rate
    b1 = b1 - db1 * learning_rate
    w2 = w2 - dw2 * learning_rate
    b2 = b2 - db2 * learning_rate

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters

def BPNN_model(X, Y, h_n, input_n, output_n, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    x_n = input_n   #输入层节点数
    y_n = output_n  #输出层节点数

    #1。初始化参数
    parameters = initialize_parameters(x_n, h_n, y_n)

    # 梯度下降循环
    for i in range(0, num_iterations):
        # 2。前向传播计算（z,a,y)
        a2, cache = forward_propagation(X, parameters)
        # 3。计算损失函数Loss Function
        cost = compute_cost(a2, Y, parameters)
        # 4。反向传播
        grads = backward_propagation(parameters, cache, X, Y)
        # 5。更新参数
        parameters = update_parameters(parameters, grads)

        #每1000次迭代，输出一次损失函数 损失函数值越小，模型越好
        if print_cost and i % 1000 == 0:
            print('迭代第%i次，代价函数为：%f' % (i, cost))

    return parameters

def load_data(vocab_dir):
    """
    返回单词以及类别id，为构建矩阵作准备
    :param vocab_dir: 词汇表
    :return:  words, word_to_id, categories, cat_to_id
    """
    words, word_to_id = read_vocab(vocab_dir)
    categories, cat_to_id = read_category()
    # vocab_size = len(words)
    return words, word_to_id, categories, cat_to_id

if __name__ == "__main__":
    test_dir = 'file/cnews/test.csv'
    vocab_dir = 'file/cnews/vocab.txt'
    train_dir = 'file/cnews/train.csv'

    words, word_to_id, categories, cat_to_id = load_data(vocab_dir)

    x_pad, y_pad = process_file(train_dir, word_to_id, cat_to_id, max_length=600)



