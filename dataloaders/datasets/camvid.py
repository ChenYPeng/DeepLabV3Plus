import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from dataloaders.mypath import get_path
from torchvision import transforms
import dataloaders.augmentation as tr


class CamVidDataSet(Dataset):

    # 构造函数
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.root = get_path(self.cfg)
        self.image_path = os.path.join(self.root, self.split + '_images')
        self.label_path = os.path.join(self.root, self.split + '_labels')
        self.images_path_list = self.read_file(self.image_path)
        self.labels_path_list = self.read_file(self.label_path)

    # 重载getitem函数，使类可以迭代
    def __getitem__(self, index):
        _image, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _image, 'target': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def __len__(self):
        return len(self.images_path_list)

    @classmethod
    def read_file(cls, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, file) for file in files_list]
        file_path_list.sort()
        return file_path_list

    def _make_img_gt_point_pair(self, index):
        image = Image.open(self.images_path_list[index]).convert('RGB')
        label = Image.open(self.labels_path_list[index])
        return image, label

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomRotate(60),
            tr.RandomResize(480, 360),
            tr.RandomHorizontalFlip(),
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

    config = get_config('camvid/deeplab-resnet-camvid.json')

    CamVid_Dataset = CamVidDataSet(config, split='train')

    dataloader = DataLoader(CamVid_Dataset, batch_size=2, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['target'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset=config['dataset'])
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
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
