import os
import json
import torch
import numpy as np

from torch.utils.data import DataLoader
from gpuinfo import GPUInfo
from dataloaders import get_test_dataset
from dataloaders.utils import get_class_name
from tools.log import get_logger
from tools.trainer.trainer import Trainer
from tools.metrics.metrics import Evaluator
from models import build_model, modified_mode_dict

np.seterr(divide='ignore', invalid='ignore')


def test(cfg, runid, use_pth='best_Eval_MIoU.pth'):
    dataset = cfg["dataset"]
    model_name = cfg["model_name"]
    class_num = cfg['num_class']
    train_logdir = os.path.join('run', dataset, model_name, runid)
    test_logdir = os.path.join('res', dataset, model_name, runid)
    logger = get_logger(test_logdir, 'test')
    logger.info(f'Conf | use logdir {train_logdir}')
    logger.info(f'Conf | use dataset {dataset}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 测试集
    test_set = get_test_dataset(cfg)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True)

    # models
    model = build_model(cfg).to(device)
    gpu_ids = GPUInfo.check_empty()

    # 多GPU测试
    if len(gpu_ids) > 1:
        new_state_dict = modified_mode_dict(os.path.join(train_logdir, use_pth))
        model.load_state_dict(new_state_dict)
    # 单GPU测试
    else:
        model.load_state_dict(torch.load(os.path.join(train_logdir, use_pth)))
    # 多GPU

    # 定义混淆矩阵
    metric = Evaluator(class_num)

    # 类别标签
    class_name = get_class_name(dataset)

    test_mat = Trainer.test(
        logger, metric, test_loader, model, device, dataset, test_logdir
    )

    # 可视化
    # Visualizer.plot_matrix('Test', test_mat, class_name, test_logdir)


if __name__ == '__main__':

    config = 'camvid/deeplab-resnet-camvid.json'
    run_id = '2022-07-21-16-29-9818'
    config_dir = os.path.join('configs', config)
    with open(config_dir, 'r') as fp:
        cfg = json.load(fp)
    
    test(cfg, run_id)
