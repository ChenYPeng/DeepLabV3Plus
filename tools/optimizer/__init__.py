from torch.optim import SGD, Adam
from tools.optimizer.ranger import Ranger
from tools.optimizer.radam import RAdam


def get_optimizer(cfg, model):
    if cfg['optimizer'] == 'Ranger':
        return Ranger(model, lr=cfg['base_lr'], weight_decay=cfg['weight_decay'])

    elif cfg['optimizer'] == 'SGD':
        return SGD(model, lr=cfg['base_lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

    elif cfg['optimizer'] == 'Adam':
        return Adam(model, lr=cfg['base_lr'], weight_decay=cfg['weight_decay'])

    elif cfg['optimizer'] == 'RAdam':
        return RAdam(model, lr=cfg['base_lr'], weight_decay=cfg['weight_decay'])

    else:
        return
