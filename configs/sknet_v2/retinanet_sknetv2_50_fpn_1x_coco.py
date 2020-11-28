_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'
model = dict(
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='SKNetV2',
        depth=50,
        # groups=32,
        # base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))

optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
