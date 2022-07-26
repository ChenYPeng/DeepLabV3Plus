from dataloaders.datasets.hanfeng import HanFengDataSet
from dataloaders.datasets.camvid import CamVidDataSet
from dataloaders.datasets.voc import VOCDataSet
from dataloaders.datasets.sbd import SBDDataSet
from dataloaders.datasets.combine_sbd import CombineSBDDataSet
from dataloaders.datasets.cityscapes import CityscapesDataSet
from torch.utils.data import DataLoader


def get_dataset(cfg):
    if cfg['dataset'] == 'hanfeng':
        train_set = HanFengDataSet(cfg, "train")
        val_set = HanFengDataSet(cfg, "val")
        test_set = HanFengDataSet(cfg, 'test')
        return train_set, val_set, test_set

    elif cfg['dataset'] == 'camvid':
        train_set = CamVidDataSet(cfg, "train")
        val_set = CamVidDataSet(cfg, "val")
        test_set = CamVidDataSet(cfg, "test")
        return train_set, val_set, test_set

    elif cfg['dataset'] == 'cityscapes':
        train_set = CityscapesDataSet(cfg, split='train')
        val_set = CityscapesDataSet(cfg, split='val')
        test_set = CityscapesDataSet(cfg, split='test')
        return train_set, val_set, test_set

    elif cfg['dataset'] == 'pascalvoc':
        train_set = VOCDataSet(cfg, year='2012', split='train')
        val_set = VOCDataSet(cfg, year='2012', split='val')
        test_set = VOCDataSet(cfg, year='2012', split='test')
        return train_set, val_set, test_set

    elif cfg['dataset'] == 'sbd':
        train_set = SBDDataSet(cfg, split='train')
        val_set = SBDDataSet(cfg, split='val')
        test_set = SBDDataSet(cfg, split='test')
        return train_set, val_set, test_set

    elif cfg['dataset'] == 'combine_sbd':
        train_set = CombineSBDDataSet(cfg, split='train')
        val_set = CombineSBDDataSet(cfg, split='val')
        test_set = CombineSBDDataSet(cfg, split='test')
        return train_set, val_set, test_set

    else:
        raise NotImplementedError


def get_train_loader(cfg, batch_size):
    train_set, val_set, _ = get_dataset(cfg)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    return train_loader, val_loader


def get_test_loader(cfg, batch_size):
    _, _, test_set = get_dataset(cfg)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                             num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)

    return test_loader
