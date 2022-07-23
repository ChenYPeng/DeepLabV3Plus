from dataloaders.datasets.hanfeng import HanFengDataSet
from dataloaders.datasets.camvid import CamVidDataSet
from dataloaders.datasets.voc import VOCDataSet
from dataloaders.datasets.sbd import SBDDataSet
from dataloaders.datasets.combine_sbd import CombineSBDDataSet
from dataloaders.datasets.cityscapes import CityscapesDataSet


def get_dataset(cfg):
    if cfg['dataset'] == 'hanfeng':
        return HanFengDataSet(cfg, "train"), HanFengDataSet(cfg, "val")
    elif cfg['dataset'] == 'camvid':
        return CamVidDataSet(cfg, "train"), CamVidDataSet(cfg, "val")
    elif cfg['dataset'] == 'cityscapes':
        return CityscapesDataSet(cfg, split='train'), CityscapesDataSet(cfg, split='val')
    
    elif cfg['dataset'] == 'pascalvoc':
        return VOCDataSet(cfg, year='2012', split='train'), VOCDataSet(cfg, year='2012', split='val')
    
    elif cfg['dataset'] == 'sbd':
        return SBDDataSet(cfg, split='train'), SBDDataSet(cfg, split='val')
    
    elif cfg['dataset'] == 'combinesbd':
        return CombineSBDDataSet(cfg, split='train'), CombineSBDDataSet(cfg, split='val')
    else:
        raise NotImplementedError
