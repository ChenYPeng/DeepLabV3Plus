import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt


def get_label_name_colors(csv_path):
    label_names = []
    label_colors = []
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for i, row in enumerate(reader):
            if i > 0:  # 跳过第一行
                label_names.append(row[0])
                label_colors.append([int(row[1]), int(row[2]), int(row[3])])
            print(label_colors)

    return label_names, label_colors


def create_label_map(label_colors, rows, cols, row_height, col_width):
    label_map = np.ones((row_height * rows, col_width * cols, 3), dtype='uint8') * 255
    cnt = 0
    for i in range(rows):  # 1st row is black = background
        for j in range(cols):
            if cnt >= len(label_colors):  # in case, num of lables < rows * cols
                break
            beg_pix = (i * row_height, j * col_width)
            end_pix = (beg_pix[0] + 20, beg_pix[1] + 20)  # 20 is color square side
            # label_map[beg_pix[0]:end_pix[0], beg_pix[1]:end_pix[1]] = label_colors[cnt][::-1]  # RGB->BGR
            label_map[beg_pix[0]:end_pix[0], beg_pix[1]:end_pix[1]] = label_colors[cnt]
            cnt += 1
    plt.imsave('label_map%dx%d.png' % (rows, cols), label_map)
    # cv2.imwrite('label_map%dx%d.png' % (rows, cols), label_map)


def plt_label_map(label_names, label_colors, rows, cols, row_height, col_width, figsize=(12, 2), fig_title='color map'):
    # create origin map
    if os.path.exists('label_map%dx%d.png' % (rows, cols)):
        os.remove('label_map%dx%d.png' % (rows, cols))
    create_label_map(label_colors, rows, cols, row_height, col_width)
    label_map = plt.imread('label_map%dx%d.png' % (rows, cols))

    # show origin map
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.title(fig_title + '\n', fontweight='black')  # 上移一段距离，哈哈
    plt.imshow(label_map)

    cnt = 0
    for i in range(rows):  # 1st row is black = background
        for j in range(cols):
            if cnt >= len(label_names):  # in case, num of lables < rows * cols
                break
            beg_pix = (j * col_width, i * row_height)  # note! (y,x)
            plt.annotate('%s' % label_names[cnt],
                         xy=beg_pix, xycoords='dataloaders', xytext=(+8, -8), textcoords='offset points',
                         color='k', fontsize=18)
            cnt += 1

    plt.show()


if __name__ == '__main__':
    # CamVid
    label_names, label_colors = get_label_name_colors(csv_path='camvid_class_dict.csv')
    plt_label_map(label_names, label_colors, rows=2, cols=6, row_height=1, col_width=3, figsize=(30, 2),
                  fig_title='CamVid-12class')
