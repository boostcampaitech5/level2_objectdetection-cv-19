# RandomCrop, HorizontalFlip, VerticalFlip, ToGray, GaussNoise, MotionBlur,
# MedianBlur, Blur, CLAHE, Sharpen, Emboss, RandomBrightnessContrast, HueSaturationValue


dataset_type = 'CocoDataset'
data_root = '/opt/ml/dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

_classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

albu_train_transforms = [
    dict(
        type='RandomSizedBBoxSafeCrop',
        height=1024,
        width=1024,
        erosion_rate=0.3,
        interpolation=1,
        p=0.3
    ),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='VerticalFlip',
                p=1
            ),
            dict(
                type='HorizontalFlip',
                p=1
            ),
        ],
        p=0.5
    ),
    dict(
        type='ToGray',
        p=0.1
    ),
    dict(
        type='GaussNoise',
        var_limit=(20, 100), 
        p = 0.3
    ),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='Blur',
                blur_limit=3,
                p=1.0
            ),
            dict(
                type='GaussianBlur',
                p=1.0
            ),
            dict(
                type='MedianBlur',
                blur_limit=5,
                p=1.0
            ),
            dict(
                type='MotionBlur',
                p = 1.0
            )
        ],
        p=0.2
    ),
    dict(
        type='CLAHE',
        p=0.3
    ),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=(0.0, 0.15),
        contrast_limit=(0.0, 0.15),
        p=0.3
    ),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=10,
        p=0.3
    )
]


train_pipeline = [
    dict(type='Mosaic', img_scale=(1024,1024), pad_val=50.0, prob=0.3),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True
        ),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024,1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'clean_30_train_fold1.json',
            img_prefix=data_root,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False,
            classes = _classes,
        ), 
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes = _classes,
        ann_file=data_root + 'val_fold1.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes = _classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
