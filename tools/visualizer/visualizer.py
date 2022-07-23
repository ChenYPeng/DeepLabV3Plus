import os
import itertools
import numpy as np
import matplotlib.pyplot as plt


class Visualizer(object):

    # 绘制混淆矩阵
    @staticmethod
    def plot_matrix(mode, cm, class_list, out_dir, cmap=plt.cm.RdYlGn):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 按行进行归一化

        plt.rc('font', family='Times New Roman', size='18')  # 设置字体样式、大小
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=800)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)  # 侧边的颜色条带

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_list, yticklabels=class_list,
               title=mode + ' Confusion Matrix',
               ylabel='True Label',
               xlabel='Pred Label')

        plt.rcParams['font.sans-serif'] = ['SimHei']  # plt.rcParams两行是用于解决标签不能显示汉字的问题
        plt.rcParams['axes.unicode_minus'] = False

        # 通过绘制格网，模拟每个单元格的边框
        ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        # 将x轴上的Label旋转45度
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fmt = '.2f'
        thresh = cm.max() / 2.
        # 把数字显示到热力图中
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            # 注意是将confusion[i][j]放到j,i这个位置
            if i == j:  # 只显示对角线上的数据
                ax.text(j, i, format(cm[i, j], fmt),
                        fontsize=15,
                        ha="center",  # 水平居中
                        va="center",  # 垂直居中
                        color="white" if cm[i, j] > thresh else "black")  # 颜色控制

        # 必选操作：显示图像。
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        plt.savefig(os.path.join(out_dir, mode + ' Confusion Matrix.png'), bbox_inches='tight', dpi=800)
        # plt.show()
        plt.close()

    @staticmethod
    def plot_curve(epoch, mode, train, valid, out_dir):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=800)
        x = list(range(epoch + 1))
        ax.plot(x, train, label="Train" + str(mode))
        ax.plot(x, valid, label="Valid" + str(mode))
        ax.set_title('Train & Valid ' + mode + ' Curve')
        plt.rc('font', family='Times New Roman', size='18')  # 设置字体样式、大小
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        plt.savefig(os.path.join(out_dir, mode + '.png'), dpi=800, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_lr(epoch, lr_list, out_dir):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=800)
        x = list(range(epoch + 1))

        ax.plot(x, lr_list)
        ax.set(title='LR Scheduler Curve',
               ylabel='lr',
               xlabel='epoch')
        plt.rc('font', family='Times New Roman', size='18')  # 设置字体样式、大小
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        plt.savefig(os.path.join(out_dir, 'lr.png'), dpi=800, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_single_curve(x, y, x_label, y_label, title, out_dir):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=800)
        ax.plot(x, y)
        ax.set(title=title, ylabel=y_label, xlabel=x_label)
        plt.rc('font', family='Times New Roman', size='18')  # 设置字体样式、大小
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        plt.savefig(os.path.join(out_dir, title + '.png'), dpi=800, bbox_inches='tight')
        plt.close()

    @staticmethod
    def data_visualize(**images):
        """PLot images in one row."""
        n = len(images)
        plt.figure(figsize=(16, 5))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        plt.show()
