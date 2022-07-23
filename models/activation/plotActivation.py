import matplotlib.pyplot as plt
import numpy as np


def Sigmoid(x):
    y = 1 / (np.exp(-x) + 1)
    return y


def Softmax(x):
    # x = x - np.max(x)
    y = np.exp(x) / np.sum(np.exp(x))
    return y


def Tanh(x):
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    # y = np.tanh(x)
    return y


def ReLU(x):
    y = np.where(x < 0, 0, x)
    return y


def LeakyReLU(x, a):
    # LeakyReLU的a参数不可训练，人为指定。
    y = np.where(x < 0, a * x, x)
    return y


def PReLU(x, a):
    # PReLU的a参数可训练
    y = np.where(x < 0, a * x, x)
    return y


def ReLU6(x):
    y = np.minimum(np.maximum(x, 0), 6)
    return y


def Swish(x, b):
    y = x * (np.exp(b * x) / (np.exp(b * x) + 1))
    return y


def Mish(x):
    # 这里的Mish已经经过e和ln的约运算
    temp = 1 + np.exp(x)
    y = x * ((temp * temp - 1) / (temp * temp + 1))
    return y


def Grad_Swish(x, b):
    y_grad = np.exp(b * x) / (1 + np.exp(b * x)) + x * (b * np.exp(b * x) / ((1 + np.exp(b * x)) * (1 + np.exp(b * x))))
    return y_grad


def Grad_Mish(x):
    temp = 1 + np.exp(x)
    y_grad = (temp * temp - 1) / (temp * temp + 1) + x * (4 * temp * (temp - 1)) / (
            (temp * temp + 1) * (temp * temp + 1))
    return y_grad


if __name__ == '__main__':
    x = np.arange(-10, 10, 0.01)
    fig, ax = plt.subplots(dpi=800)

    # 指定坐标的位置
    # 设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')  # 隐藏掉右边框线
    ax.spines['top'].set_color('none')  # 隐藏掉左边框线

    # 设置坐标轴位置
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # 绑定坐标轴位置，data为根据数据自己判断
    ax.spines['bottom'].set_position(('dataloaders', 0))  # data表示通过值来设置x轴的位置，将x轴绑定在y=0的位置
    ax.spines['left'].set_position(('axes', 0.5))  # axes表示以百分比的形式设置轴的位置，即将y轴绑定在x轴50%的位置，也就是x轴的中点

    ax.plot(x, Sigmoid(x), label='Sigmoid')
    ax.plot(x, ReLU(x), label='ReLU')
    ax.plot(x, ReLU6(x), label='ReLU6')
    ax.plot(x, Swish(x, 1), label='Swish')
    ax.plot(x, Mish(x), label='Mish')

    ax.set_title('Activation Functions')
    ax.legend()

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig('../result/activation.png')
    plt.show()
