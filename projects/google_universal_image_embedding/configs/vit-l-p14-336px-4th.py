_base_ = [
    './_base_/inshop_bs32_336.py',
    'mmcls::_base_/schedules/cub_bs64.py',
    'mmcls::_base_/default_runtime.py',
]

model = dict(
    type='ImageToImageRetriever',
    image_encoder=[
        dict(
            type='VisionTransformer',
            arch='l',
            img_size=336,
            patch_size=14,
            drop_rate=0.1,
            output_patch_token=False,
            pre_norm=True),
        dict(type='CLIPProjection', in_channels=1024, out_channels=768),
        dict(
            type='LinearReduction',
            in_channels=768,
            out_channels=512,
            norm_cfg=None,
            bias=False),
    ],
    head=dict(
        type='ArcFaceClsHead',
        num_classes=10000,  # num_classes is dummy for inference
        in_channels=512,
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

load_from = 'https://github.com/okotaku/clshub-weights/releases/download/v0.1.1guie/ViT-L-14-336-4th.pth'  # noqa
