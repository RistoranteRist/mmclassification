_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cub_bs8_448.py',
    '../_base_/schedules/cub_bs64.py',
    '../_base_/default_runtime.py',
]

# model settings
# use pre-train weight from https://github.com/facebookresearch/vicregl # noqa
pretrained = 'https://dl.fbaipublicfiles.com/vicregl/resnet50_alpha0.75.pth'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=pretrained)),
    head=dict(num_classes=200, ))

# schedule settings
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=5e-4,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999))
)

# runtime settings
default_hooks = dict(logger=dict(type='LoggerHook', interval=20))
