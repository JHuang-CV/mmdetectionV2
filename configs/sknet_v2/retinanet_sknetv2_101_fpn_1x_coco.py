_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='SKNetV2',
        depth=101,
        # groups=32,
        # base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))

optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
#resume_from = './work_dirs/retinanet_x101_at_max_fpn_1x_coco/epoch_4.pth'
