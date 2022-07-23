from tools.loss.loss import SegmentationLosses


# 损失函数 & 类别权重平衡
def get_loss(cfg, weight=None):
    return {
        # mode = 'ce' or 'focal'
        'Cross': SegmentationLosses(weight=weight, cuda=True).build_loss(mode='focal')

    }[cfg['criterion']]

