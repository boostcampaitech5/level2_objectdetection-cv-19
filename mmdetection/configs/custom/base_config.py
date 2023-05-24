import os
from mmdet.utils import get_device

_base_ = [
    "../../_base_/models/faster_rcnn_r50_fpn.py",
    "../../_base_/datasets/coco_trash_detection.py",
    "../../_base_/schedules/schedule_2x.py",
    "../../_base_/default_trash_runtime.py",
]
# schedule_adamw_cosine

exp_name = "MM_baseline_fine_tuning"
worker = "jisu"

# Set folder path if resume with custom pre-trained weights
weight_name = "MM_baseline_train40_res1024"
weight_dir = os.path.join("/opt/ml/output/", weight_name)

batch_size = 4
max_epochs = 24
device = get_device()

work_dir = os.path.join("/opt/ml/output/", exp_name)
os.makedirs(work_dir, exist_ok=True)

train_annotation = "clean_40_train_fold1.json"
val_annotation = "val_fold1.json"

model=dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=10)
        )
    )

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="MMDetWandbHook",
            interval=100,
            init_kwargs=dict(entity="cv-19", project="mmdetection", name=exp_name),
            by_epoch=True,
            num_eval_images=100,
            log_checkpoint=False,
            log_checkpoint_metadata=False,
        )
    ],
)

evaluation = dict(interval=1, save_best='bbox_mAP_50')