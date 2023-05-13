checkpoint_config = dict(max_keep_ckpts=1, interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="MMDetWandbHook",
            interval=100,
            init_kwargs=dict(entity="vip_cv19", project="mmdetection", name="exp"),
            by_epoch=True,
            num_eval_images=100,
            # 평가에 사용된 총 이미지 수. 값이 0이면 validation_data_path에 있는 모든 이미지가 평가에 사용.
            log_checkpoint=False,
            log_checkpoint_metadata=False,
        )
    ],
)
# yapf:enable
custom_hooks = [dict(type="NumClassCheckHook")]

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1), ("val", 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = "fork"

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
