# 모듈 import
#import wandb

#from mmcv import Config
#from mmdet.utils import get_device
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from custom.custom_config import TrainCustomConfig

## train의 기능 3가지(dataset load, model load, model train) 구현
def train():
    # wandb init 기능
    # wandb.init(entity="vip_cv19", project="object_detection", config=vars(cfg))

    # Custom config 로드
    cfg = TrainCustomConfig()

    # dataset 로드
    datasets = [build_dataset(cfg.data.train)]

    # model 로드
    model = build_detector(cfg.model)
    model.init_weights()

    # model 학습
    train_detector(model, datasets[0], cfg, distributed=False, validate=False)

if __name__=="__main__":
    train()