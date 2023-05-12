import os

_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_trash_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_trash_runtime.py'
]

# optimizer
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)


exp_name = "MM_retinanet_r18_fpn_1x_train35"
worker = "jisu"

batch_size = 4
max_epochs = 50

work_dir = os.path.join("/opt/ml/output/", exp_name)
os.makedirs(work_dir, exist_ok=True)

train_annotation = "clean_35_train_fold1.json"
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

evaluation = dict(interval=1, save_best='bbox_mAP', metric='bbox') # bbox_loss