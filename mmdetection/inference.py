# 모듈 import
import os
import argparse

import mmcv
import numpy as np
import pandas as pd

from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from pandas import DataFrame
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    
    #MM_baseline_train30
    parser.add_argument("--cfg_folder", type=str, default="exp_name", help="cfg folder")
    parser.add_argument("--seed", type=int, default=2023, help="random seed")

    args = parser.parse_args()

    return args


def inference():
    args = parse_args()
    
    # Custom config 로드
    base_path = '/opt/ml/output'
    exp_path = os.path.join(base_path, args.cfg_folder)
    dir_lst = os.listdir(exp_path)
    
    cfg_file, weight_file = None, None
    for i in dir_lst:
        if i.endswith('.py'):
            cfg_file = i
        if i.startswith('best'):
            weight_file = i
            
    if cfg_file==None:
        print('There is no cfg file!')
        print(f'your path: {exp_path}\nfile list: {dir_lst}')
        return
    if weight_file==None:
        weight_file = 'latest.pth'

    #print(cfg_file, weight_file)
    cfg_path = os.path.join(exp_path, cfg_file)
    cfg = Config.fromfile(cfg_path)

    # dataset config 수정
    cfg.data.test.test_mode = True

    cfg.seed=2021
    cfg.gpu_ids = [1]

    cfg.model.train_cfg = None

    # dataset & dataloader 로드
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False
    )

    # model 로드, checkpoint 로드
    checkpoint_path = os.path.join(cfg.work_dir, weight_file)

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))  # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")  # ckpt load

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    # output 계산
    output = single_gpu_test(model, data_loader, show_score_thr=0.05)

    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ""
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += (
                    str(j)
                    + " "
                    + str(o[4])
                    + " "
                    + str(o[0])
                    + " "
                    + str(o[1])
                    + " "
                    + str(o[2])
                    + " "
                    + str(o[3])
                    + " "
                )

        prediction_strings.append(prediction_string)
        file_names.append(image_info["file_name"])

    # 최종 submission 파일 생성
    submission = pd.DataFrame()
    submission["PredictionString"] = prediction_strings
    submission["image_id"] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f"submission_{cfg.exp_name}.csv"), index=None)


'''
python inference.py --cfg_folder MM_baseline_train30
'''
if __name__ == "__main__":
    inference()
