import torch
import numpy as np


def get_class_name(dataset):
    if dataset == 'cityscapes':
        class_name = get_cityscapes_class()
    elif dataset == 'camvid':
        class_name = get_camvid_class()
    elif dataset == 'hanfeng':
        class_name = get_hanfeng_class()
    elif dataset == 'combine_sbd' or 'pascal':
        class_name = get_pascal_class()
    else:
        raise NotImplementedError
    return class_name


def decode_seg_map_sequence(label_masks, dataset):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


# 灰度图像与RGB图像的互相转换
def decode_segmap(label_mask, dataset, plot=False):
    if dataset == 'pascalvoc' or dataset == 'coco' or dataset == 'sbd' or dataset == 'combine_sbd':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    elif dataset == 'camvid':
        n_classes = 12
        label_colours = get_camvid_labels()
    elif dataset == 'hanfeng':
        n_classes = 3
        label_colours = get_hanfeng_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()  # h x w
    g = label_mask.copy()  # h x w
    b = label_mask.copy()  # h x w
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))  # 生成h x w x 3的矩阵，因为mask只有RGB三个通道

    # 矩阵每一行的3个数（0，models1，2），分别填入R，G，B的数值
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask, dataset):
    if dataset == 'pascal' or dataset == 'sbd' or dataset == 'combine_sbd':
        label_class = get_pascal_labels()
    elif dataset == 'cityscapes':
        label_class = get_cityscapes_labels()
    elif dataset == 'camvid':
        label_class = get_camvid_labels()
    else:
        raise NotImplementedError

    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(label_class):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_cityscapes_class():
    return ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky', 'person',
            'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


def get_cityscapes_labels():
    return np.asarray([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                       [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35],
                       [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                       [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])


def get_pascal_class():
    return ['Unlabelled', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def get_pascal_labels():
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def get_camvid_class():
    # 天空、建筑物、柱子、道路、人行道、树、标志符号、栅栏、汽车、行人、自行车、其他
    return ['SignSymbol', 'Sky', 'Building', 'Pole', 'Road', 'SideWalk',
            'Tree', 'Unlabelled', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']


def get_camvid_labels():
    return np.asarray(
        [[192, 128, 128], [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128], [0, 0, 192],
         [128, 128, 0], [0, 0, 0], [64, 64, 128], [64, 0, 128], [64, 64, 0], [0, 128, 192]])


def get_hanfeng_class():
    # Unlabelled、A、B、C、D
    # return ['Unlabelled', 'A', 'B', 'C', 'D']
    return ['Unlabelled', 'A', 'B']


def get_hanfeng_labels():
    # return np.asarray(
    #     [[0, 0, 0], [192, 128, 128], [128, 128, 128], [128, 0, 0], [192, 192, 128]])
    return np.asarray(
        [[0, 0, 0], [192, 128, 128], [128, 128, 128]])


def denormalize_image(image):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    image *= std
    image += mean

    return image
