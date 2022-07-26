import os


def get_path(cfg):
    root = 'E:/datasets'
    if cfg['dataset'] == 'hanfeng':
        return os.path.join(root, 'hanfengSeg/')  # folder that contains HanFeng/.
    elif cfg['dataset'] == 'camvid':
        return os.path.join(root, 'camvid/')  # folder that contains CamVid/.
    elif cfg['dataset'] == 'cityscapes':
        return os.path.join(root, 'Cityscapes/')  # folder that contains leftImg8bit/
    elif cfg['dataset'] == 'pascalvoc':
        return os.path.join(root, 'VOCdevkit/')  # folder that contains VOCdevkit/.
    elif cfg['dataset'] == 'sbd':
        return os.path.join(root, 'benchmark_RELEASE/')  # folder that contains dataset/.
    elif cfg['dataset'] == 'combine_sbd':
        return os.path.join(root, 'combine_sbd/')  # folder that contains dataset/.
    elif cfg['dataset'] == 'coco':
        return os.path.join(root, 'coco/')
    else:
        print('Dataset {} not available.'.format(cfg['dataset'])
