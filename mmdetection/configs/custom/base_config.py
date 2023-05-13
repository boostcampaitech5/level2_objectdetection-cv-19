_base_ = [
    "../_base_/models/faster_rcnn_r50_fpn.py",
    "../_base_/datasets/coco_trash_detection.py",
    "../_base_/schedules/schedule_adamw_cosine.py",
    "../_base_/default_trash_runtime.py",
]

exp_name = "MM_base_config"
worker = "jisu"

batch_size = 4
max_epochs = 20

work_dir = "/opt/ml"
train_annotation = "clean_40_train_fold1.json"
val_annotation = "val_fold1.json"
