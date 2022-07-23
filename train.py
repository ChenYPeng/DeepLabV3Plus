import os
import time
import torch
import numpy as np
import torch.nn as nn
from gpuinfo import GPUInfo
from models import build_model
from torch.utils.data import DataLoader

from dataloaders import get_dataset
from dataloaders.utils import get_class_name
from dataloaders.mypath import get_path
from tools.loss import get_loss
from tools.trainer.trainer import Trainer
from tools.optimizer import get_optimizer
from tools.scheduler import get_scheduler
from tools.metrics.metrics import Evaluator
from tools.utils.calculate_weights import calculate_weigths_labels
from models.sync_batchnorm.replicate import patch_replication_callback

np.seterr(divide='ignore', invalid='ignore')


def run(cfg, logger):
    # 判断是否多gpu训练
    gpu_ids = GPUInfo.check_empty()

    # 加载模型
    model = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids).to(device)
        patch_replication_callback(model)
        model = model.to(device)
    else:
        model = model.to(device)

    # 同时获取训练集、验证集和测试集
    train_data, eval_data = get_dataset(cfg)
    class_num = cfg['num_class']
    class_name = get_class_name(cfg["dataset"])
    batch_size = cfg['batch_size'] * len(gpu_ids)
    train_loader = DataLoader(train_data, batch_size, shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False,
                             num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)

    # 损失函数 & 类别权重平衡
    if cfg["use_balanced_weights"]:
        classes_weights_path = os.path.join(get_path(cfg), cfg["dataset"] + '_classes_weights.npy')
        if os.path.isfile(classes_weights_path):
            weight = np.load(classes_weights_path)
        else:
            weight = calculate_weigths_labels(get_path(cfg), cfg["dataset"], train_loader, cfg["num_class"])
        weight = torch.from_numpy(weight.astype(np.float32))
    else:
        weight = None

    # 定义超参数
    criterion = get_loss(cfg, weight=weight)
    # 优化器
    optimizer = get_optimizer(cfg, model.parameters())
    # 学习率衰减
    scheduler = get_scheduler(cfg, len(train_loader))
    # 定义混淆矩阵
    metric = Evaluator(class_num)
    # 训练次数
    max_epoch = cfg["epochs"]

    # 打印参数
    logger.info(f'Conf | use model {cfg["model_name"]}')
    logger.info(f'Conf | use dataset {cfg["dataset"]}')
    logger.info(f'Conf | use criterion function {cfg["criterion"]}')
    logger.info(f'Conf | use optimizer function {cfg["optimizer"]}')
    logger.info(f'Conf | use scheduler function {cfg["scheduler"]}')
    logger.info(f'Conf | use epochs {cfg["epochs"]}')
    logger.info(f'Conf | use crop_size {cfg["crop_size"]}')
    logger.info(f'Conf | use GPU {gpu_ids}')
    logger.info(f'Conf | use batch_size {batch_size}')

    best_pred = 0.0
    lr_list = []
    train_loss_list, train_acc_list, train_miou_list = [], [], []
    eval_loss_list, eval_acc_list, eval_miou_list = [], [], []

    start = time.time()

    for epoch in range(max_epoch):

        train_loss = Trainer.train(
            logger, metric, train_loader, model, criterion, optimizer, scheduler, epoch, max_epoch, device, best_pred)
        eval_loss, eval_acc, eval_miou, eval_mat = Trainer.valid(
            logger, metric, eval_loader, model, criterion, epoch, max_epoch, device)

        # 保存数据
        # train_loss_list.append(train_loss)
        # eval_loss_list.append(eval_loss)
        # train_miou_list.append(train_miou)
        # eval_miou_list.append(eval_miou)
        # train_acc_list.append(train_acc)
        # eval_acc_list.append(eval_acc)
        # lr_list.append(optimizer.param_groups[0]['lr'])

        # 可视化
        # Visualizer.plot_curve(epoch, 'Loss', train_loss_list, eval_loss_list, out_dir=cfg['log_dir'])
        # Visualizer.plot_matrix('Train', train_mat, class_name, out_dir=cfg['log_dir'])
        # Visualizer.plot_matrix('Eval', eval_mat, class_name, out_dir=cfg['log_dir'])
        # Visualizer.plot_curve(epoch, 'MIoU', train_miou_list, eval_miou_list, out_dir=cfg['log_dir'])
        # Visualizer.plot_curve(epoch, 'Acc', train_acc_list, eval_acc_list, out_dir=cfg['log_dir'])
        # Visualizer.plot_lr(epoch, lr_list, out_dir=cfg['log_dir'])

        # 保存验证集最好的模型
        new_pred = eval_miou
        if new_pred > best_pred:
            best_pred = new_pred
            torch.save(model.state_dict(), os.path.join(cfg['log_dir'], 'best_Eval_MIoU.pth'))

    end = time.time()

    # 验证集最高MIoU
    logger.info(f'Conf | Best MIoU {best_pred}')
    logger.info(f'Conf | Run Time {(end - start) / 3600}')


if __name__ == '__main__':
    from tools.log import get_logdir, get_logger

    config = 'camvid/deeplab-resnet-camvid.json'
    cfg, log_dir = get_logdir(config)
    logger = get_logger(log_dir, 'train')
    logger.info(f'Conf | use log_dir {log_dir}')
    cfg['log_dir'] = log_dir
    run(cfg, logger)
