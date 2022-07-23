import numbers

import torch
import random
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


# numpy 转 tensor
class ToTensor(object):
    """
    Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    """

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        img = sample['image']
        mask = sample['target']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img, 'target': mask}


# 归一化
class Normalize(object):
    """
    Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['target']

        # 正则化，x=(x-mean)/std
        # 只对image正则化, image [C,H,W]tensor RGB -1.0~1.0
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        # 先转为ndarray，再转为tensor，不归一化，维度保持不变
        # label [H,W]tensor trainId
        mask = np.array(mask).astype(np.float32)

        return {'image': img, 'target': mask}


# 随机尺寸缩放
class RandomResize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, sample):
        img = sample['image']
        mask = sample['target']

        img = img.resize((self.height, self.width), Image.BILINEAR)
        mask = mask.resize((self.height, self.width), Image.NEAREST)  # label要用邻近差值

        return {'image': img, 'target': mask}


# 按比例缩放
class ScaleResize(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        img = sample['image']
        mask = sample['target']

        assert img.size == mask.size
        h, w = img.size

        img = img.resize((int(h * self.scale), int(w * self.scale)), Image.BILINEAR)
        mask = mask.resize((int(h * self.scale), int(w * self.scale)), Image.NEAREST)  # label要用邻近差值

        return {'image': img, 'target': mask}


# 随机比例缩放
class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['target']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img, 'target': mask}


# 填充缩放
class FixedResize(object):
    def __init__(self, height, width):  # size: (h, w)
        self.width = width
        self.height = height

    def __call__(self, sample):
        img = sample['image']
        mask = sample['target']

        assert img.size == mask.size

        img = img.resize((self.height, self.width), Image.BILINEAR)
        mask = mask.resize((self.height, self.width), Image.NEAREST)  # label要用邻近差值

        return {'image': img, 'target': mask}


# 随机旋转
class RandomRotate(object):
    def __init__(self, degree=60, p=0.5):
        self.degree = degree
        self.p = p
        self.random_angle = np.random.randint(-1 * self.degree, self.degree)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['target']
        if random.random() < self.p:  # p的概率会翻转
            img = img.rotate(self.random_angle, Image.BILINEAR)
            mask = mask.rotate(self.random_angle, Image.NEAREST)  # label要用邻近差值

        return {'image': img, 'target': mask}


# 水平翻转
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img = sample['image']
        mask = sample['target']

        if random.random() < self.p:  # p的概率会翻转
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img, 'target': mask}


# 垂直翻转
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img = sample['image']
        mask = sample['target']

        if random.uniform(0, 1) < self.p:  # 50%的概率会翻转
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': img, 'target': mask}


# 高斯模糊
class RandomGaussianBlur(object):
    def __init__(self, radius=1, p=0.5):
        self.radius = radius
        self.p = p

    def __call__(self, sample):
        img = sample['image']
        mask = sample['target']

        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.randint(0, self.radius)))  # radius-模糊半径

        return {'image': img, 'target': mask}


# 调整明亮度
class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, sample):

        img = sample['image']
        mask = sample['target']

        transforms = []

        if self.brightness is not None:
            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            transforms.append(Lambda(lambda img: TF.adjust_brightness(img, brightness_factor)))

        if self.contrast is not None:
            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            transforms.append(Lambda(lambda img: TF.adjust_contrast(img, contrast_factor)))

        if self.saturation is not None:
            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            transforms.append(Lambda(lambda img: TF.adjust_saturation(img, saturation_factor)))

        if self.hue is not None:
            hue_factor = random.uniform(self.hue[0], self.hue[1])
            transforms.append(Lambda(lambda img: TF.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)

        for transform in transforms:
            img = transform(img)

        return {'image': img, 'target': mask}


# 填充中心裁剪
class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['target']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img, 'target': mask}


# 随机裁剪
class RandomCrop(object):
    """
    Crop the given PIL Image at a random location.
    自定义实现图像与label随机裁剪相同的位置
    没办法直接使用transform.resize() 因为是像素级别的标注，而resize会将这些标注变成小数
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        img = sample['image']
        lbl = sample['target']

        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s' % (img.size, lbl.size)
        if self.padding > 0:
            img = TF.pad(img, self.padding)
            lbl = TF.pad(lbl, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = TF.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            lbl = TF.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = TF.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
            lbl = TF.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)

        img = TF.crop(img, i, j, h, w)
        label = TF.crop(lbl, i, j, h, w)

        return {'image': img, 'target': label}


class RandomColor(object):
    def __init__(self):
        super(RandomColor, self).__init__()
        pass

    def __call__(self, sample):
        """
        调整亮度、对比度、饱和度
        只调整image，不调整label
        :param image: [H,W,C] PIL Image RGB 0~255
        :param label: [H,W] PIL Image trainId
        :return: [H,W,C] PIL Image RGB 0~255,  [H,W] PIL Image trainId
        """
        img = sample['image']
        mask = sample['target']

        random_factor = np.random.randint(5, 15) / 10.  # 随机因子
        color_image = ImageEnhance.Color(img).enhance(random_factor)  # 调整图像的饱和度

        random_factor = np.random.randint(8, 13) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度

        random_factor = np.random.randint(7, 17) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度

        random_factor = np.random.randint(5, 16) / 10.  # 随机因子
        contrast_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

        return {'image': contrast_image, 'target': mask}


class PairRandomCutout(object):
    def __init__(self, mask_size=64, value=0):
        """
        按照固定大小，随机遮挡图像中的某一块方形区域
        :param mask_size: 被遮挡的区域大小，默认64x64
        :param value: 被遮挡的部分用value值填充
        """
        super(PairRandomCutout, self).__init__()
        self.mask_size = mask_size
        self.value = value

    def __call__(self, sample):
        """
        按照固定大小，随机遮挡图像中的某一块方形区域
        :param image: [C,H,W] tensor，必须是tensor
        :param label: [H,W] tensor，必须是tensor
        :return: [C,H,W] tensor,  [H,W] tensor
        """
        image = sample['image']
        label = sample['target']

        _, h, w = image.shape
        top = random.randint(0, h - self.mask_size)  # 随机到遮挡部分的top
        left = random.randint(0, w - self.mask_size)  # 随机到遮挡部分的left
        if random.uniform(0, 1) < 0.5:  # 随机遮挡
            image = TF.erase(image, top, left, self.mask_size, self.mask_size,
                             v=self.value, inplace=True)

        return {'image': image, 'target': label}


class PairRandomGaussianBlur(object):
    def __init__(self):
        """
        随机高斯模糊
        """
        super(PairRandomGaussianBlur, self).__init__()
        pass

    def __call__(self, sample):
        """
        随机高斯模糊
        :param image: [H,W,C] PIL Image RGB 0~255
        :param label: [H,W] PIL Image trainId
        :return: [H,W,C] PIL Image RGB 0~255,  [H,W] PIL Image trainId
        """
        image = sample['image']
        label = sample['target']

        high = min(image.size[0], image.size[1])  # 取图像HW最小值
        radius = random.randint(1, high)  # 随机高斯模糊半径
        gaussian_filter = ImageFilter.GaussianBlur(radius=radius)  # 高斯模糊过滤器

        image = image.filter(gaussian_filter)

        return {'image': image, 'target': label}
