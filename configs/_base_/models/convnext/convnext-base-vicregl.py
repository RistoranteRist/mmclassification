_base_ = 'convnext-base.py'
model = dict(
    backbone=dict(
        layer_scale_init_value=0
    ))
