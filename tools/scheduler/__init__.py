from tools.scheduler.lr_scheduler import LR_Scheduler


def get_scheduler(cfg, len_train_loader):

    if cfg['scheduler'] == "poly":
        return LR_Scheduler("poly", cfg["base_lr"], cfg["epochs"], iters_per_epoch=len_train_loader)

    else:
        raise NotImplementedError
