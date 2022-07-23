import os
import sys
import time
import json
import shutil
import random
import logging


def get_logdir(cfg_name):
    base_dir = 'E:/Project/SemanticSegmentation/'
    cfg_dir = base_dir + 'configs/'
    run_dir = base_dir + 'run/'
    config = cfg_dir + cfg_name
    with open(config, 'r') as fp:
        cfg = json.load(fp)
    directory = os.path.join(run_dir, cfg["dataset"], cfg["model_name"])
    run_id = str(time.strftime("%Y-%m-%d-%H-%M")) + '-' + str(random.randint(1000, 10000))
    log_dir = os.path.join(directory, run_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    shutil.copy(config, log_dir)
    return cfg, log_dir


def get_logger(log_dir, mode):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Python time strftime() 返回以可读字符串表示的当地时间，格式由参数format决定
    if mode == 'train':
        log_name = f'train-{time.strftime("%Y-%m-%d-%H-%M")}.log'
    elif mode == 'test':
        log_name = f'test-{time.strftime("%Y-%m-%d-%H-%M")}.log'
    elif mode == 'pred':
        log_name = f'pred-{time.strftime("%Y-%m-%d-%H-%M")}.log'

    else:
        raise NotImplementedError
    log_file = os.path.join(log_dir, log_name)

    # create log
    logger = logging.getLogger('train')  # log初始化
    logger.setLevel(logging.INFO)  # 设置log级别

    # Formatter 设置日志输出格式
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # StreamHandler 日志输出1 -> 到控制台
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler 日志输出2 -> 保存到文件log_file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
