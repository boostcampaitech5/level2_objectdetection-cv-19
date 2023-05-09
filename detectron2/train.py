#!/usr/bin/env python
# coding: utf-8

import os
import torch
import wandb
import argparse
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.utils.util import ensure_dir
from detectron2 import model_zoo 
from detectron2.config import get_cfg

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2.model_zoo import get_config
from detectron2.train.my_trainer import MyTrainer


def RegisterDataset(data_path='../../dataset'):
    try:
        register_coco_instances('coco_trash_train', {}, os.path.join(data_path, 'train.json'), data_path)
    except AssertionError:
        pass
    
    try:
        register_coco_instances('coco_trash_test', {}, os.path.join(data_path, 'test.json'), data_path)
    except AssertionError:
        pass

    MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                            "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]


    return


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def train(cfg):
    # TODO : logging되는 config 수정
    wandb.init(entity="vip_cv19", project="object_detection", config=vars(args))
    
    RegisterDataset()
    
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument("--config", type=str, default="my_configs/example_cfg.yaml", help="config file directory address")
    parser.add_argument("--train_json", type=str, default="my_configs/example_cfg.yaml", help="config file directory address")
    parser.add_argument("--config", type=str, default="my_configs/example_cfg.yaml", help="config file directory address")
    parser.add_argument("--config", type=str, default="my_configs/example_cfg.yaml", help="config file directory address")
    parser.add_argument("--config", type=str, default="my_configs/example_cfg.yaml", help="config file directory address")
    parser.add_argument("--config", type=str, default="my_configs/example_cfg.yaml", help="config file directory address")
    parser.add_argument("--config", type=str, default="my_configs/example_cfg.yaml", help="config file directory address")

    args = parser.parse_args()

    # load config
    cfg = setup(args)
    #cfg = get_config(args.config)
    ensure_dir(cfg.OUTPUT_DIR)

    # train
    train(cfg)
    
    # TODO : CV strategy와도 연관지어서 val dataset 나누기 (현재는 따로 val 과정이 없음) - maybe stratified kfold
    # TODO : 