import os
import json
import torch
import numpy as np

from gpuinfo import GPUInfo

from tools.log import get_logger
from dataloaders.utils import get_class_name
from dataloaders import get_test_loader
from models import build_model, modified_mode_dict
from tools.trainer.trainer import Trainer
from tools.metrics.metrics import Evaluator
from tools.visualizer.visualizer import Visualizer

np.seterr(divide='ignore', invalid='ignore')


def test(cfg, id, use_pth='best_Eval_MIoU.pth'):
    dataset = cfg["dataset"]
    model_name = cfg["model_name"]
    class_num = cfg['num_class']
    train_logdir = os.path.join('run', dataset, model_name, id)
    test_logdir = os.path.join('res', dataset, model_name, id)
    logger = get_logger(test_logdir, 'test')
    logger.info(f'Conf | use logdir {train_logdir}')
    logger.info(f'Conf | use dataset {dataset}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_loader = get_test_loader(cfg, batch_size=1)

    model = build_model(cfg).to(device)
    gpu_ids = GPUInfo.check_empty()

    # 多GPU测试
    if len(gpu_ids) > 1:
        new_state_dict = modified_mode_dict(os.path.join(train_logdir, use_pth))
        model.load_state_dict(new_state_dict)
    # 单GPU测试
    else:
        model.load_state_dict(torch.load(os.path.join(train_logdir, use_pth)))

    model = model.eval()

    # 定义混淆矩阵
    metric = Evaluator(class_num)

    # 类别标签
    class_name = get_class_name(dataset)

    test_mat = Trainer.test(
        logger, metric, test_loader, model, device, dataset, test_logdir
    )


if __name__ == '__main__':
    cfg_path = 'hanfeng/deeplab-resnet-hanfeng.json'
    run_id = '2022-07-26-08-29-2265'
    config_dir = os.path.join('configs', cfg_path)
    with open(config_dir, 'r') as fp:
        config = json.load(fp)

    test(config, run_id)
