_base_ = [
    './_base_/inshop_bs32_224.py',
    'mmcls::_base_/schedules/cub_bs64.py',
    'mmcls::_base_/default_runtime.py',
]

model = dict(
    type='ImageToImageRetriever',
    image_encoder=[
        dict(
            type='VisionTransformer',
            arch='b',
            img_size=224,
            patch_size=16,
            drop_rate=0.1,
            output_patch_token=False,
            pre_norm=True),
    ],
    head=dict(
        type='ArcFaceClsHead',
        num_classes=10000,  # num_classes is dummy for inference
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=None),
    prototype={{_base_.gallery_dataloader}})

# runtime settings
default_hooks = dict(
    # log every 20 intervals
    logger=dict(type='LoggerHook', interval=20))

custom_hooks = [
    dict(type='PrepareProtoBeforeValLoopHook'),
    dict(type='SyncBuffersHook')
]

load_from = 'https://github.com/okotaku/clshub-weights/releases/download/v0.1.1guie/clip-vit-base-p16_openai-pre_3rdparty_in1k.pth'  # noqa
