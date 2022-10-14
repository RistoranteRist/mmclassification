_base_ = 'convnext-small.py'
model = dict(
    backbone=dict(
        layer_scale_init_value=0
    ))
