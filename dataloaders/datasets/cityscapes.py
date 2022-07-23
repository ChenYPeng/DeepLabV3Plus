import os
import numpy as np
from PIL import Image
from torch.utils import data

from torchvision import transforms
import dataloaders.augmentation as tr
from dataloaders.mypath import get_path


class CityscapesDataSet(data.Dataset):
    NUM_CLASSES = 19

    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.root = get_path(self.cfg)

        self.files = {}

        # images_base：原图所在目录。 annotations_base: label图所在目录。
        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        # 获取image_base目录（包括所有子目录）中，所有以.png结尾的文件；
        # 返回的结果是所有文件的‘路径+文件名’组成的list列表。
        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        # 在cityscapes数据集中包含的所有类CLASSES中，加一个'unlabelled'类。共20个类。
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        # 设置ignore_index为255，这个变量是用来干嘛的，目前还不知道。
        self.ignore_index = 255
        # 将valid_classes中19个数字为key，[0,models1,2,...,18]中19个数字为value。一一对应组成字典
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))
        # 如果self.files[split]为空，即split文件下没有任何数据，则报错。
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        # print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        # 返回该数据集里包含的所有元素个数。
        return len(self.files[self.split])

    def __getitem__(self, index):
        # 获取第index个图像的路径。rstrip()函数用来删除末尾空格。
        img_path = self.files[self.split][index].rstrip()
        # 获取第index个图像对应的label图像的路径。
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')
        # 打开原始图像，转换为RGB格式。
        _img = Image.open(img_path).convert('RGB')
        # 获取对应的label图像，用np.unint8格式打开，保存为格式np.array。
        # 之所以要有这一步，是为了下面一行的encode_segmap做准备。
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        # 转换label图像中，每一个像素的类别，便于训练。
        _tmp = self.encode_segmap(_tmp)
        # 将array转换成image图像。
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'target': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

        return sample

    def encode_segmap(self, mask):
        # Put all void classes to zero
        # 将属于void_classes的像素类别转成ignore_index。
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        # 将属于valid_classes的像素类别转成class_map对应的元素值。
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    # 提取rootdir目录及其递归子目录下，所有以suffix结尾的文件。
    def recursive_glob(self, rootdir='.', suffix=''):
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    # 数据预处理
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomScaleCrop(base_size=self.cfg['base_size'], crop_size=self.cfg['crop_size'], fill=0),
            tr.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from configs import get_config
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    config = get_config('cityscapes/deeplab-resnet-cityscapes.json')

    cityscapes_train = CityscapesDataSet(config, split='val')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()  # from torch convert to numpy n x c x h x w
            gt = sample['target'].numpy()  # from torch convert to numpy n x c x h x w
            tmp = np.array(gt[jj]).astype(np.uint8)  # tmp.shape=c x h x w
            segmap = decode_segmap(tmp, dataset=config['dataset'])
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])  # img_tmp=h x w x c
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
