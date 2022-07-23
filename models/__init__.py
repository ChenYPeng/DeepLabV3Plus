import torch
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.deeplabv3p import DeepLabV3Plus

BatchNorm = SynchronizedBatchNorm2d


def build_model(cfg):
    return {
        'DeepLabV3Plus': DeepLabV3Plus,

    }[cfg['model_name']](backbone=cfg["backbone"], num_class=cfg['num_class'])


def modified_mode_dict(model_path):
    state_dict = torch.load(model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def load_pretrained_model(model_dict, pretrained_dict):
    # 将pretrained_dict里不属于model_dict的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 更新现有的model_dict
    model_dict.update(pretrained_dict)

    return model_dict
