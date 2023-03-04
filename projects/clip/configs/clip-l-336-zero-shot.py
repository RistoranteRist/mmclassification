_base_ = [
    'mmcls::_base_/datasets/imagenet_bs64_clip_336.py',
    'mmcls::_base_/schedules/imagenet_bs4096_AdamW.py',
    'mmcls::_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['projects.clip.clip'], allow_failed_imports=False)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='l',
        img_size=336,
        patch_size=14,
        drop_rate=0.1,
        avg_token=True,
        output_cls_token=False,
        pre_norm=True,
        final_norm=False,
    ),
    neck=dict(
        type='CLIPProjection',
        in_channels=1024,
        out_channels=768,
    ),
    head=dict(type='ZeroShotClsHead'))

custom_hooks = [dict(type='BuildClassVocabHook')]

val_dataloader = dict(batch_size=4, num_workers=2)

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
