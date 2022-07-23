import json
import os

print(os.getcwd())


def get_config(cfg_name):
    base_dir = 'E:/Project/SemanticSegmentation/'
    cfg_dir = base_dir + 'configs/'
    config = cfg_dir + cfg_name
    with open(config, 'r') as fp:
        cfg = json.load(fp)
    return cfg
