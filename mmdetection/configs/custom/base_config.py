import os

_base_ = [
    "../_base_/models/faster_rcnn_r50_fpn.py",
    "../_base_/datasets/coco_trash_detection.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_trash_runtime.py",
]
# schedule_adamw_cosine

exp_name = "MM_base_config"
worker = "jisu"

batch_size = 4
max_epochs = 50

work_dir = os.path.join("/opt/ml/output/", exp_name)
os.makedirs(work_dir, exist_ok=True)

train_annotation = "clean_40_train_fold1.json"
val_annotation = "val_fold1.json"

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="MMDetWandbHook",
            interval=100,
            init_kwargs=dict(entity="vip_cv19", project="object_detection", name=exp_name),
            by_epoch=True,
            num_eval_images=100,
            log_checkpoint=False,
            log_checkpoint_metadata=False,
        )
    ],
)

evaluation = dict(interval=1, save_best='bbox_mAP', metric='bbox') # bbox_loss, bbox_mAP