import numpy as np

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\\L    P     N
 P      TP    FP
 N      FN    TN
"""


class Evaluator(object):
    def __init__(self, num_classes):
        # 数据集类别数量
        self.num_classes = num_classes
        # 初始化混淆矩阵 (空)
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    # 计算混淆矩阵
    def gen_confusion_matrix(self, pre_labels, gt_labels, ignore_labels):
        """
        :param pre_labels
        :param gt_labels
        :return: confusion_matrix
        """

        # 验证标签取值是对的 通过mask可以把为true的元素挑出来，并且拉成一行
        mask = (gt_labels >= 0) & (gt_labels < self.num_classes)
        for IgLabel in ignore_labels:
            mask &= (gt_labels != IgLabel)
        label = self.num_classes * gt_labels[mask].astype('int') + pre_labels[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)

        return confusion_matrix

    # 正确像素所占的比例
    def pixel_accuracy(self):
        # PA =  (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    # CPA 计算每个类被正确分类的像素的比例 Recall
    def pixel_accuracy_class(self):
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    # MPA 均像素精度 计算每个类被正确分类的像素的比例， 之后求所有类的平均
    def mean_pixel_accuracy(self):
        # 横着代表预测值，竖着代表真实值
        classAcc = self.pixel_accuracy_class()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def Precision(self):
        precesion_arr = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        return precesion_arr

    # 各个类别的IoU
    def intersection_over_union(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)  # 取对角元素的值，返回列表
        # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(
            self.confusion_matrix)
        Iou = intersection / union  # 返回列表，其值为各个类别的IoU

        return Iou

    # mIoU 平均交并比
    def mean_intersection_over_union(self):
        IoU = self.intersection_over_union()
        MIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return MIoU

    # FWIOU 权频交并比 根据每个类出现的频率为其设置权重
    def frequency_weighted_intersection_over_union(self):
        # FWIOU = [(TP+FN)/(TP+FP+TN+FN)] * [TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = self.intersection_over_union()

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def kappa(self):
        pe_rows = np.sum(self.confusion_matrix, axis=0)
        pe_cols = np.sum(self.confusion_matrix, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
        # np.trace求矩阵的迹，即对角线的和
        po = np.trace(self.confusion_matrix) / float(sum_total)
        return (po - pe) / (1 - pe)

    # 添加数据
    def add_batch(self, pre_labels, gt_labels, ignore_labels):
        assert pre_labels.shape == gt_labels.shape
        self.confusion_matrix += self.gen_confusion_matrix(pre_labels, gt_labels, ignore_labels)  # 得到混淆矩阵
        return self.confusion_matrix

    # 重置矩阵
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def get_results(self):
        cm = self.confusion_matrix
        acc = self.pixel_accuracy_class()
        acc_cls = self.mean_pixel_accuracy()
        iou = self.intersection_over_union()
        mean_iou = self.mean_intersection_over_union()
        FWIoU = self.frequency_weighted_intersection_over_union()
        cls_iou = dict(zip(range(self.num_classes), iou))

        return {"Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": FWIoU,
                "Mean IoU": mean_iou,
                "Class IoU": cls_iou, }


# 测试内容
if __name__ == '__main__':
    imgPredict = np.array([[0, 1, 2], [2, 1, 1]])  # 可直接换成预测图片
    imgLabel = np.array([[0, 1, 255], [1, 1, 2]])  # 可直接换成标注图片
    ignore_labels = [255]
    metric = Evaluator(3)  # 3表示有3个分类，有几个分类就填几, 0也是1个分类
    hist = metric.add_batch(imgPredict, imgLabel, ignore_labels)
    pa = metric.pixel_accuracy()
    cpa = metric.pixel_accuracy_class()
    mpa = metric.mean_pixel_accuracy()
    IoU = metric.intersection_over_union()
    mIoU = metric.mean_intersection_over_union()
    print('hist is :\n', hist)
    print('PA is : %f' % pa)
    print('cPA is :', cpa)  # 列表
    print('mPA is : %f' % mpa)
    print('IoU is : ', IoU)
    print('mIoU is : ', mIoU)
