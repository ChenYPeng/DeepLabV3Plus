import os
import json
import tqdm
import torch
import glob

import numpy as np
import torchvision.transforms as T

from PIL import Image
from gpuinfo import GPUInfo

from dataloaders.utils import decode_segmap
from models import build_model, modified_mode_dict

np.seterr(divide='ignore', invalid='ignore')


def predict(cfg, runid, use_pth='best_Eval_MIoU.pth'):
    dataset = cfg["dataset"]
    model_name = cfg["model_name"]
    train_logdir = os.path.join('run', dataset, model_name, runid)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 加载测试集
    image_files = []
    input_path = 'samples/src'
    output_path = 'samples/res'
    if os.path.isdir(input_path):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG', 'bmp']:
            files = glob.glob(os.path.join(input_path, '**/*.%s' % ext), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(input_path):
        image_files.append(input_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

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

    for img_path in tqdm.tqdm(image_files):
        ext = os.path.basename(img_path).split('.')[-1]
        img_name = os.path.basename(img_path)[:-len(ext) - 1]
        img = Image.open(img_path).convert('RGB')

        img = transform(img).unsqueeze(0)  # To tensor of N C H W
        img = img.to(device)

        with torch.no_grad():
            outputs = model(img)
        pred = outputs.max(1)[1].squeeze().cpu().data.numpy()  # (1024, 1280)
        pred = decode_segmap(pred, dataset, plot=False)  # 保存预测标签图需注释此行
        pred = Image.fromarray(np.uint8(pred))
        pred.save(os.path.join(output_path, img_name + '.png'))


if __name__ == '__main__':
    config = 'hanfeng/deeplab-resnet-hanfeng.json'
    run_id = '2022-07-26-08-29-2265'
    config_dir = os.path.join('configs', config)
    with open(config_dir, 'r') as fp:
        cfg = json.load(fp)

    predict(cfg, run_id)
