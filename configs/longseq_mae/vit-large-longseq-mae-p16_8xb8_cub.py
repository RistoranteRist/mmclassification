_base_ = [
    '../_base_/models/vit-large-p16.py', '../_base_/datasets/cub_bs8_448.py',
    '../_base_/schedules/cub_bs64.py', '../_base_/default_runtime.py'
]
fp16 = dict(loss_scale='dynamic')

# model settings
checkpoint = 'vitl_dec512d16h8b_1600ep_img448_crop0.2-1.0_maskds2.pth'  # noqa
model = dict(
    type='ImageClassifier',
    backbone=dict(
        img_size=448,
        with_cp=True,
        avg_token=True,
        output_cls_token=False,
        final_norm=False,
        init_cfg=dict(
            _delete_=True,
            type='Pretrained',
            checkpoint=checkpoint,
            prefix='backbone')),
    head=dict(
        _delete_=True,
        type='LinearClsHead',
        num_classes=200,
        in_channels=1024,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False))

# schedule settings
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=5e-6,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=5.0),
)

default_hooks = dict(
    # log every 20 intervals
    logger=dict(type='LoggerHook', interval=20),
    # save last three checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))
