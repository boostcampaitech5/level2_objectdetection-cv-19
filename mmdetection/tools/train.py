# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import sys

sys.path.insert(0, "/opt/ml/level2_objectdetection-cv-19-develop/mmdetection")
import mmcv
from mmcv.runner import load_checkpoint
from mmcv import Config
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (
    collect_env,
    get_device,
    get_root_logger,
    replace_cfg_vals,
    setup_multi_processes,
    update_data_root,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")

    parser.add_argument("config", help="train config file path")

    parser.add_argument("--seed", type=int, default=2023, help="random seed")

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)
    # set multi-process settings
    setup_multi_processes(cfg)

    cfg.runner.max_epochs = cfg.max_epochs
    cfg.data.samples_per_gpu = cfg.batch_size

    cfg.gpu_ids = [0]
    distributed = False

    cfg.data.train.ann_file = cfg.data_root + cfg.train_annotation
    cfg.data.val.ann_file = cfg.data_root + cfg.val_annotation

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text
    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    cfg.device = get_device()

    # set random seeds
    seed = args.seed
    logger.info(f"Set random seed to {seed}, " f"deterministic: {args.deterministic}")
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta["seed"] = seed
    meta["exp_name"] = osp.basename(args.config)

    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        assert "val" in [mode for (mode, _) in cfg.workflow]
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # checkpoint를 얼마나 저장할 것인가
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES)

    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    if cfg.weight_dir:
        dir_lst = os.listdir(cfg.weight_dir)
        weight_file = None
        for i in dir_lst:
            if i.startswith('best'):
                weight_file = i

        if weight_file==None:
            weight_file = 'latest.pth'
        weight_path = os.path.join(cfg.weight_dir, weight_file)
        checkpoint = load_checkpoint(model, weight_path, map_location="cpu")  # ckpt load
    else:
        model.init_weights()

    model.CLASSES = datasets[0].CLASSES

    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        timestamp=timestamp,
        meta=meta,
    )

'''
python tools/train.py configs/custom/MM_baseline_cascade_train35.py
'''
if __name__ == "__main__":
    main()
