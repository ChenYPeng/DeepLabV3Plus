import os
import shutil
import json
import time
import random


class Saver(object):
    @staticmethod
    def make_logdir(config):
        with open(config, 'r') as fp:
            cfg = json.load(fp)
        directory = os.path.join('E:/SemanticSegmentation/run', cfg["dataset"], cfg["model_name"])
        run_id = str(time.strftime("%Y-%m-%d-%H-%M")) + '-' + str(random.randint(1000, 10000))
        log_dir = os.path.join(directory, 'experiment-{}'.format(run_id))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        shutil.copy(config, log_dir)
        return cfg, log_dir


if __name__ == '__main__':
    fun = Saver.make_logdir(r"E:/SemanticSegmentation/cfg/deeplab-resnet-camvid.json")
