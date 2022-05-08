# -*- coding: utf-8 -*-


import numpy as np
import random
import matplotlib.pyplot as plt


def relu(x, deriv=False):
    """
    激活函数
    :param x: 传入参数
    :param deriv: 是否求导的标志
    :return: 激活函数/求导后结果
    """
    if deriv:
        x[x < 0] = 0
        x[x >= 0] = 1
        return x
    else:
        return np.maximum(0, x)


def loss(y_pre, y):
    """
    计算损失函数
    :param y_pre: 预测得到的值
    :param y: 真实数据的值
    :return: loss
    """
    loss = np.square(y_pre - y).sum() / 2
    return loss


def predict(x, w):
    """
    前向传播
    :param x: 传入数据
    :param w: 系数矩阵
    :return: 前向传播结果
    """
    layer1_in = np.dot(x, w[0].T)
    layer1_out = relu(layer1_in)
    layer2 = np.dot(layer1_out, w[1].T)
    return layer2


def backforward(x, y, w, lr):
    """
    反向传播
    :param x: 输入矩阵
    :param y: 真实矩阵
    :param w: 系数矩阵
    :param lr: 学习率
    :return: 新的系数矩阵
    """
    x = x.reshape(1, 1000)
    layer1_in = np.dot(w[0], x.T)
    layer1_out = relu(layer1_in)
    layer2 = np.dot(w[1], layer1_out)
    error2 = layer2 - y
    error1 = np.dot(w[1].T, error2)
    w2_d = np.dot(error2, layer1_out.T)
    w1_d = np.dot(np.multiply(error1, relu(layer1_out, deriv=True)), x)
    w[0] += - lr * w1_d
    w[1] += - lr * w2_d
    return w


def train(x, y, w, lr, epoch, batch, iteration, W, b):
    """
    训练模型
    :param x: 输入矩阵
    :param y: 真实矩阵
    :param w: 系数矩阵
    :param lr: 学习率
    :param epoch: 训练次数
    :param batch: 批大小
    :param iteration: 迭代次数
    :param W: 系数矩阵
    :param b: 偏置项
    :return: loss 系数矩阵 迭代次数
    """
    l = []
    num = []
    for i in range(epoch):
        for j in range(1, iteration + 1):
            x_list = list(x)
            x_ = random.sample(x_list, batch)
            x_ = np.array(x_)
            for k in range(batch):
                y_ = (np.dot(x_[k], W).T + b).reshape(10, 1)
                w = backforward(x_[k], y_, w, lr)
            y_out = predict(x, w)
            loss1 = loss(y_out, y) / len(x)
            if j % 10 == 0:
                l.append(loss1)
                num.append(j)
                print("迭代了%d次：loss值是：%.8f" % (j, loss1))
            if loss1 < 1e-6:
                iters = j
                break
        if loss1 >= 1e-6:
            iters = j
    return l, w, iters, num


if __name__ == '__main__':
    lr = 0.00003
    epoch = 1
    batch = 64
    iteration = 1000
    x = np.random.randn(100, 1000)
    print("X.len:", len(x))
    W = np.random.normal(0, 1, (1000, 10))
    b = 1.0
    print("x：", x.shape)
    y = np.dot(x, W) + b
    print("y：", y.shape)
    w1 = np.random.normal(0, 1, (100, 1000))
    w2 = np.random.normal(0, 1, (10, 100))
    print("w2: ", w2.shape)
    w = [w1, w2]
    y1 = predict(x, w)
    loss_reg = loss(y1, y) / len(x)
    print("loss_reg:", loss_reg)
    l, w, j, num = train(x, y, w, lr, epoch, batch, iteration, W, b)
    print("迭代了%d次：loss值是：%.8f" % (j, l[-1]))

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(num, l)
    plt.xlabel('迭代次数')
    plt.ylabel('loss')
    plt.show()

