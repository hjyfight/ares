"""
VOC2007 Faster R-CNN evaluation configuration.
This config is for diagnosing the extremely low mAP issue.
"""

_base_ = './base.py'

batch_size = 2

# Model and weights
detector = dict(
    cfg_file='work_dirs/faster_rcnn_r50_fpn_1x_voc0712/faster_rcnn_r50_fpn_1x_voc0712.py',
    weight_file='work_dirs/faster_rcnn_r50_fpn_1x_voc0712/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth'
)

# Attack configuration (for eval_only mode, attack won't be applied)
attack_method = dict(
    type='pgd',
    kwargs=dict(
        eps=3/255,
        norm='inf'
    )
)

# Image saving options
adv_image = dict(save=False, save_folder='adv_images', with_bboxes=True)
clean_image = dict(save=True, save_folder='clean_images', with_bboxes=True)
