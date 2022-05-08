'''
learn_rate设置得非常小 所以跑起来很慢 但是调大了又容易梯度施加得太快直接梯度爆炸
learn_rate设置得非常小 所以跑起来很慢 但是调大了又容易梯度施加得太快直接梯度爆炸
learn_rate设置得非常小 所以跑起来很慢 但是调大了又容易梯度施加得太快直接梯度爆炸
'''
import random
import torch
import numpy as np
D_in = 1000
H = 100
D_out = 10
batch_size = 64


def relu(x, deriv=False):
    """
    激活函数
    """
    if deriv:
        b = x.clone().detach().reshape(-1)
        for i in range(len(b)):
            if b[i] >= 0:
                b[i] = 1
            else:
                b[i] = 0
        return b.reshape(x.shape)
    else:
        return np.maximum(0, x)


def synthetic_data(w1, w2, b1, b2, num_examples):  # @save
    """生成原始数据"""
    X = torch.normal(0, 1, (num_examples, D_in))
    y1 = relu(torch.matmul(X, w1) + b1)
    y2 = torch.matmul(y1, w2) + b2
    y2 += torch.normal(0, 0.01, y2.shape)
    return X, y2


w1 = torch.normal(0, 1, (D_in, H))
w2 = torch.normal(0, 1, (H, D_out))
b1 = torch.normal(0, 1, (1, H))
b2 = torch.normal(0, 1, (1, D_out))
features, ys = synthetic_data(w1, w2, b1, b2, 10000)

print('features:', features.shape, '\nys:', ys.shape)


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


w1p = torch.normal(0, 1, (D_in, H))
w2p = torch.normal(0, 1, (H, D_out))
b1p = torch.normal(0, 1, (1, H))
b2p = torch.normal(0, 1, (1, D_out))


def linreg(X, w1, w2, b1, b2):  # @save
    """线性回归模型"""
    y1 = relu(torch.matmul(X, w1) + b1)
    y2 = torch.matmul(y1, w2) + b2
    return y2


def squared_loss(y_hat, y):  # @save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):  # @save
    """小批量随机梯度下降"""
    # with torch.no_grad():
    for param in params:
        param -= lr * param._grad__.sum(dim=0) / batch_size
            # param.grad.zero_()


lr = 0.000003
num_epochs = 500
net = linreg


def loss(w1, w2, b1, b2, X, y):
    t1 = torch.matmul(X, w1)
    y1 = (t1+b1)
    t3 = relu(y1)
    t4 = torch.matmul(t3, w2)
    yp = t4+b2
    loss = (yp-y)**2
    b2._grad__ = 2*(yp-y)
    w2._grad__ = torch.matmul(t3.T, b2._grad__)/50000
    b1._grad__ = torch.matmul(b2._grad__, w2.T)*relu(y1, deriv=True)
    w1._grad__ = torch.matmul(X.T, b1._grad__)
    return loss

print('start training')
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, ys):
        loss(w1p, w2p, b1p, b2p, X, y)
        sgd([w1p, w2p, b1p, b2p], lr, batch_size)  # 使用参数的梯度更新参数
    # with torch.no_grad():
    train_l = squared_loss(net(features, w1p, w2p, b1p, b2p), ys)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# print(f'w1的估计误差: {w1p - w1}')
# print(f'b1的估计误差: {b1p - b1}')
# print(f'w2的估计误差: {w2p - w2}')
# print(f'b2的估计误差: {b2p - b2}')

# print(w1[0],w1p[0])
