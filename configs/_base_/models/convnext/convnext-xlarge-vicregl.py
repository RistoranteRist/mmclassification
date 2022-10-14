_base_ = 'convnext-xlarge.py'
model = dict(
    backbone=dict(
        layer_scale_init_value=0
    ))
