from models import (
    SCFlowRefiner, RAFTEncoder, SCFlowDecoder, SequenceLoss,
    MultiClassPoseHead, RAFTLoss, DisentanglePointMatchingLoss, 
    L1Loss, TensorboardImgLoggerHook
)
from mmengine.hooks import (CheckpointHook, LoggerHook)
from datasets import (LoadImages, LoadMasks, PoseJitter, 
                      ComputeBbox, Crop, RandomHSV, 
                      RandomNoise, RandomSmooth, Resize, 
                      Pad, RemapPose, Normalize, ToTensor, 
                      Collect)
from mmengine.dataset import DefaultSampler
from mmengine.dataset import ConcatDataset
from datasets import LUMIPianoRefineDataset, LUMIPianoSuperviseTrainDataset
from metrics import LUMIPianoADD

default_scope = None
dataset_root = 'data/lumi_piano_dataset'
syn_dataset_root = 'data/syn_lumi_piano_dataset'

CLASS_NAMES = ['lumi_piano']
label_obj_id_map={"LUMI": 0}
symmetry_types = {}

normalize_mean = [0., 0., 0., ]
normalize_std = [255., 255., 255.]
image_scale = 256
mesh_diameter = [314.086245]
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type=LoadImages, color_type='unchanged', file_client_args=file_client_args),
    dict(type=LoadMasks, file_client_args=file_client_args),
    dict(type=PoseJitter,
        jitter_angle_dis=(0, 15),
        jitter_x_dis=(0, 15),
        jitter_y_dis=(0, 15),
        jitter_z_dis=(0, 50),
        angle_limit=45, 
        translation_limit=200,
        add_limit=1.,
        mesh_dir=dataset_root + '/models_1080_mm_eval',
        mesh_diameter=mesh_diameter,
        jitter_pose_field=['gt_rotations', 'gt_translations'],
        jittered_pose_field=['ref_rotations', 'ref_translations']),
    dict(type=ComputeBbox, mesh_dir=dataset_root + '/models_1080_mm_eval', clip_border=False),
    dict(type=Crop,
        size_range=(1.0, 1.25), 
        crop_bbox_field='ref_bboxes',
        clip_border=False,
        pad_val=128,
    ),
    #dict(type='RandomBackground', background_dir='data/coco', p=0.3, file_client_args=file_client_args),
    dict(type=RandomHSV, h_ratio=0.2, s_ratio=0.5, v_ratio=0.5),
    dict(type=RandomNoise, noise_ratio=0.1),
    dict(type=RandomSmooth, max_kernel_size=5.),
    dict(type=Resize, img_scale=image_scale, keep_ratio=True),
    dict(type=Pad, size=(image_scale, image_scale), center=True, pad_val=dict(img=(128, 128, 128), mask=0)),
    dict(type=RemapPose, keep_intrinsic=False),
    dict(type=Normalize, mean=normalize_mean, std=normalize_std, to_rgb=True),
    dict(type=ToTensor, stack_keys=[], ),
    dict(type=Collect, 
        annot_keys=[
            'ref_rotations', 'ref_translations', 
            'gt_rotations', 'gt_translations', 'gt_masks',
            'init_add_error', 'init_rot_error', 'init_trans_error',
            'k', 'labels'],
        meta_keys=(
            'img_path', 'ori_shape', 'ori_k',
            'img_shape', 'img_norm_cfg', 
            'scale_factor', 'transform_matrix',
            'ori_gt_rotations', 'ori_gt_translations'),
    ),
]

test_pipeline = [
    dict(type=LoadImages, color_type='unchanged', file_client_args=file_client_args),
    dict(type=ComputeBbox, mesh_dir=dataset_root + '/models_1080_mm_eval', clip_border=False, filter_invalid=False),
    dict(type=Crop, size_range=(1.1, 1.1), crop_bbox_field='gt_bboxes',  clip_border=False, pad_val=128), #change back
    dict(type=Resize, img_scale=image_scale, keep_ratio=True),
    dict(type=Pad, size=(image_scale, image_scale), center=True, pad_val=dict(img=(128, 128, 128), mask=0)),
    dict(type=RemapPose, keep_intrinsic=False),
    dict(type=Normalize, mean=normalize_mean, std=normalize_std, to_rgb=True),
    dict(type=ToTensor, stack_keys=[], ),
    dict(type=Collect, 
        annot_keys=[
            'ref_rotations', 'ref_translations',
            'gt_rotations', 'gt_translations',
            'labels','k', 'ori_k', 'transform_matrix', 
        ],
        meta_keys=(
            'img_path', 'ori_shape', 'img_shape', 'img_norm_cfg', 
            'scale_factor',  'keypoints_3d', 'geometry_transform_mode'),
    ),
]

real_piano_set = dict(
    type=LUMIPianoSuperviseTrainDataset,
    data_root=dataset_root,
    track_start=1,
    track_end=7,
    pipeline=train_pipeline,
    keypoints_json=dataset_root + '/keypoints/bbox.json',
    keypoints_num=8,
    class_names=CLASS_NAMES,
    sample_num=-1,
    min_visib_fract=0.1,
    mesh_symmetry_types=symmetry_types,
    mesh_diameter=mesh_diameter,
    meshes_eval=dataset_root + '/models_1080_mm_eval',
    track_prefix="track_",
    num_digit=2,
    annot_prefix='lumi_piano',
    metainfo={"class_names": CLASS_NAMES}
)
syn_piano_set = dict(
    type=LUMIPianoSuperviseTrainDataset,
    data_root=syn_dataset_root,
    track_start=2,
    track_end=4,
    pipeline=train_pipeline,
    keypoints_json=syn_dataset_root + '/keypoints/bbox.json',
    keypoints_num=8,
    class_names=CLASS_NAMES,
    sample_num=-1,
    min_visib_fract=0.1,
    mesh_symmetry_types=symmetry_types,
    mesh_diameter=mesh_diameter,
    meshes_eval=syn_dataset_root + '/models_1080_mm_eval',
    track_prefix="synthetic_data",
    num_digit=1,
    annot_prefix='syn_lumi_piano',
    metainfo={"class_names": CLASS_NAMES}
)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=ConcatDataset, 
        datasets=[real_piano_set, syn_piano_set]
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=LUMIPianoRefineDataset,
        data_root=dataset_root,
        track_start=1,
        track_end=7,
        ref_annots_root=dataset_root + '/initial_pose',
        keypoints_json=dataset_root + '/keypoints/bbox.json',
        pipeline=test_pipeline,
        class_names=CLASS_NAMES,
        keypoints_num=8,
        mesh_symmetry=symmetry_types,
        meshes_eval=dataset_root+'/models_1080_mm_eval',
        mesh_diameter=mesh_diameter
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=LUMIPianoRefineDataset,
        data_root=dataset_root,
        track_start=1,
        track_end=7,
        ref_annots_root=dataset_root + '/initial_pose',
        keypoints_json=dataset_root + '/keypoints/bbox.json',
        pipeline=test_pipeline,
        class_names=CLASS_NAMES,
        keypoints_num=8,
        mesh_symmetry=symmetry_types,
        meshes_eval=dataset_root+'/models_1080_mm_eval',
        mesh_diameter=mesh_diameter,
    ),
)

model = dict(
    type=SCFlowRefiner,
    cxt_channels=128,
    h_channels=128,
    seperate_encoder=False,
    max_flow=400.,
    filter_invalid_flow=True,
    encoder=dict(
        type=RAFTEncoder,
        in_channels=3,
        out_channels=256,
        net_type='Basic',
        norm_cfg=dict(type='IN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['InstanceNorm2d'], val=1, bias=0)
        ]),
    cxt_encoder=dict(
        type=RAFTEncoder,
        in_channels=3,
        out_channels=256,
        net_type='Basic',
        norm_cfg=dict(type='BN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['SyncBatchNorm2d'], val=1, bias=0)
        ]),
    decoder=dict(
        type=SCFlowDecoder,
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=8,
        detach_flow=True,
        detach_mask=True,
        detach_pose=True,
        detach_depth_for_xy=True,
        mask_flow=False,
        mask_corr=False,
        pose_head_cfg=dict(
            type=MultiClassPoseHead,
            num_class=21,
            in_channels=224,
            net_type='Basic',
            rotation_mode='ortho6d',
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            act_cfg=dict(type='ReLU'),
        ),
        corr_lookup_cfg=dict(align_corners=True),
        gru_type='SeqConv',
        act_cfg=dict(type='ReLU')),
    flow_loss_cfg=dict(
        type=SequenceLoss,
        gamma=0.8,
        loss_func_cfg=dict(
            type=RAFTLoss,
            loss_weight=.1,
            max_flow=400.,
        )
    ),
    pose_loss_cfg=dict(
        type=SequenceLoss,
        gamma=0.8,
        loss_func_cfg=dict(
            type=DisentanglePointMatchingLoss,
            symmetry_types=symmetry_types,
            mesh_diameter=mesh_diameter,
            mesh_path=dataset_root+'/models_1080_mm_eval',
            loss_type='l1',
            disentangle_z=True,
            loss_weight=10.0,
        )
    ),
    mask_loss_cfg=dict(
        type=SequenceLoss,
        gamma=0.8,
        loss_func_cfg=dict(
            type=L1Loss,
            loss_weight=10.,
        )
    ),
    renderer=dict(
        mesh_dir=dataset_root + '/models_1080_mm',
        image_size=(image_scale, image_scale),
        shader_type='Phong',
        soft_blending=False,
        render_mask=False,
        render_image=True,
        seperate_lights=True,
        faces_per_pixel=1,
        blur_radius=0.,
        sigma=1e-12,
        gamma=1e-12,
        background_color=(.5, .5, .5),
        obj_label_in_file=False,
        label_obj_id_map=label_obj_id_map,
        mesh_ext='obj'
    ),
    freeze_bn=False,
    freeze_encoder=False,
    train_cfg=dict(),
    test_cfg=dict(iters=8),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='/home/giakhang/dev/SCFlow/work_dirs/scflow_lumi_piano_real/iter_100000.pth'
    ),
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0001,
        amsgrad=False
    ),
    clip_grad=dict(max_norm=10.)
)

param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=0.0004,
        total_steps=100,
        pct_start=0.05,
        anneal_strategy='linear'
    ),
]

val_evaluator = dict(
    type=LUMIPianoADD,
    data_root=dataset_root,
    track_start=1,
    track_end=7,
    ref_annots_root=dataset_root + '/initial_pose',
    keypoints_json=dataset_root + '/keypoints/bbox.json',
    class_names=CLASS_NAMES,
    keypoints_num=8,
    mesh_symmetry=symmetry_types,
    meshes_eval=dataset_root+'/models_1080_mm_eval',
    mesh_diameter=mesh_diameter,
    metrics={'auc':[], 'add':[0.05, 0.10, 0.20, 0.50]},
    save_dir='work_dirs/scflow_lumi_piano_real_mix_syn_2025_03_26/val_results',
)
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

num_gpus = 1
default_hooks = dict(
    checkpoint=dict(type=CheckpointHook, interval=10000, by_epoch=False),
    logger=dict(type=LoggerHook, interval=50),
    visualization=dict(type=TensorboardImgLoggerHook, interval=100, image_format='HWC')
)
work_dir = 'work_dirs/scflow_lumi_piano_real_mix_syn_2025_03_26'
load_from = '/home/giakhang/dev/SCFlow/work_dirs/scflow_lumi_piano_real/iter_100000.pth'

resume = False