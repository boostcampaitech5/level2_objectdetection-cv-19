import os
import yaml
import wandb
import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")

    # parser.add_argument("--dataset_cfg", type=str, default="recycle.yaml", help="dataset config file path")
    parser.add_argument("--train_cfg", type=str, default="default.yaml", help="train config file path")
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--weight", type=str, default="yolov8n.pt", help="yolov8 model weight(.pt) or path(outputs/~/best.pt)")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # cfg_path = os.path.join("./ultralytics/datasets", args.dataset_cfg)
    custom_cfg_path = os.path.join("./ultralytics/yolo/cfg", args.train_cfg)
    with open(custom_cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize wandb
    # wandb.init(entity="vip_cv19", project="object_detection", config=cfg)

    # Define model
    # init에서 주어진 string 가지고 _load 수행, model
    model = YOLO(args.weight)

    # Set the logger for the model
    model.logger = wandb

    # Train the model
    # default : {'task': 'detect', 'data': 'coco.yaml', 'imgsz': 640, 'single_cls': False}
    # train과 함께 주어진 인자를 덮어쓰기 하는 방식
    model.train(cfg=custom_cfg_path)

    # Finish wandb logging
    wandb.finish()


"""
pretrained model load: python train.py --weight yolov8l.pt --train_cfg V1_yolov8l_pt_img1024.yaml
best.pt부터 이어서 load: python train.py --weight outputs/V1_yolov8l_pt_img1024/weights/best.pt --train_cfg V1_yolov8l_pt_img1024.yaml
"""
if __name__ == "__main__":
    main()
