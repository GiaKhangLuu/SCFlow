from typing import Optional, Sequence
import itertools
import mmcv
import glob
import trimesh
import mmengine
import numpy as np
from os import path as osp
from pathlib import Path
from registry import DATASETS
from terminaltables import AsciiTable
from torch.utils.data import Dataset
import random
from tqdm import tqdm

from .pipelines import Compose

@DATASETS.register_module()
class LUMIPianoRefineDataset(Dataset):
    def __init__(
        self,
        data_root,
        track_start,
        track_end,
        pipeline:Sequence[dict],
        ref_annots_root:str,
        keypoints_json:str,
        keypoints_num:int,
        gt_annots_root:str=None,
        filter_invalid_pose:bool=False,
        depth_range: Optional[tuple]=None,
        class_names : Optional[tuple]=None,
        label_mapping: dict = None,
        target_label: list = None,
        meshes_eval: str = None,
        mesh_symmetry: dict = {},
        mesh_diameter: list = []
    ):
        super().__init__()
        self.data_root = data_root
        self.keypoints_num = keypoints_num
        self.class_names = class_names
        self.label_mapping = label_mapping
        self.target_label = target_label
        self.mesh_symmetry_types = mesh_symmetry
        self.mesh_diameter = np.array(mesh_diameter)
        self.track_start = track_start
        self.track_end = track_end

        if meshes_eval is not None:
            self.meshes = self._load_mesh(meshes_eval, ext='.obj')
        else:
            self.meshes = None

        if pipeline is not None:
            self.transformer = Compose(pipeline)

        self.keypoints_3d = self._load_keypoints_3d(keypoints_json)

        self.ref_annots_root = ref_annots_root
        self.gt_annots_root = data_root if gt_annots_root is None else gt_annots_root
        self.mask_path_tmpl = "data/{}/mask_visib/{:05}_{:05}.png"
        self.filter_invalid_pose = filter_invalid_pose
        self.depth_range = depth_range
        self.gt_seq_pose_annots, self.ref_seq_pose_annots, self.img_files = self._load_data()

    def _load_mesh(self, mesh_path, ext='.ply'):
        if osp.isdir(mesh_path):
            mesh_paths = glob.glob(osp.join(mesh_path, '*'+ext))
            mesh_paths = sorted(mesh_paths)
        else:
            mesh_paths = [mesh_path]
        meshs = [trimesh.load(p) for p in mesh_paths]
        return meshs
    
    def _load_keypoints_3d(self, keypoints_json):
        keypoints_3d = mmengine.load(keypoints_json)
        keypoints_3d = np.array(keypoints_3d, dtype=np.float32).reshape(-1, self.keypoints_num, 3)
        return keypoints_3d  
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        results = self.getitem(index)
        while results is None:
            index = random.randint(0, len(self.img_files) - 1)
            results = self.getitem(index)
        return results

    def _load_data(self):
        ann_files=[
            osp.join(self.data_root, f"data/track_{t:02d}/image_set/{self.class_names[0]}_test.txt")
            for t in range(self.track_start, self.track_end + 1) 
        ]
        image_prefixes=[
            osp.join(self.data_root, f"data/track_{t:02d}")
            for t in range(self.track_start, self.track_end + 1)
        ]
    
        assert len(ann_files) == len(image_prefixes)
        img_files = []
        gt_seq_pose_annots = dict()
        ref_seq_pose_annots = dict()

        for ann_file, scene_root in zip(ann_files, image_prefixes):
            with open(str(ann_file), 'r') as f_ann:
                indices = [line.strip("\r\n") for line in f_ann.readlines()]  # string ids

            for im_id in tqdm(indices):
                int_im_id = int(im_id)
                rgb_path = osp.join(scene_root, "rgb/{:05d}.png").format(int_im_id)
                assert osp.exists(rgb_path), rgb_path
                img_files.append(rgb_path) 

            scene = scene_root.split("/")[-1]

            ref_pose_json_path = osp.join(self.ref_annots_root, scene, "scene_gt.json")
            ref_info_json_path = osp.join(self.ref_annots_root, scene, "scene_gt_info.json")  # bbox_obj, bbox_visib
            gt_pose_json_path = osp.join(scene_root, "scene_gt.json")
            gt_info_json_path = osp.join(scene_root, "scene_gt_info.json")  # bbox_obj, bbox_visib
            camera_json_path = osp.join(scene_root, "scene_camera.json")

            gt_pose_annots = mmengine.load(gt_pose_json_path)
            gt_infos = mmengine.load(gt_info_json_path)
            camera_annots = mmengine.load(camera_json_path)
            ref_pose_annots = mmengine.load(ref_pose_json_path)
            ref_infos = mmengine.load(ref_info_json_path)

            gt_seq_pose_annots[scene] = dict(
                pose=gt_pose_annots, 
                camera=camera_annots, 
                gt_info=gt_infos
            )
            ref_seq_pose_annots[scene] = dict(
                pose=ref_pose_annots, 
                camera=camera_annots
            )
            if Path(ref_info_json_path).exists():
                ref_seq_pose_annots[scene].update(
                    ref_info=ref_infos)
        
        return gt_seq_pose_annots, ref_seq_pose_annots, img_files
    
    def getitem(self, index):
        img_path = self.img_files[index]
        _,  seq_name, _, img_name = img_path.rsplit('/', 3)
        img_id = int(osp.splitext(img_name)[0])
        
        gt_seq_annots, ref_seq_annots = self.gt_seq_pose_annots[seq_name], self.ref_seq_pose_annots[seq_name]

        # load ground truth pose annots
        if str(img_id) in gt_seq_annots['pose']:
            gt_pose_annots = gt_seq_annots['pose'][str(img_id)]
        else:
            gt_pose_annots = gt_seq_annots['pose']["{:06}".format(img_id)]
        
        # load referece pose annots
        if str(img_id) in ref_seq_annots['pose']:
            ref_pose_annots = ref_seq_annots['pose'][str(img_id)]
        else:
            ref_pose_annots = ref_seq_annots['pose']["{:06}".format(img_id)]
        
        # load camera intrisic
        if str(img_id) in gt_seq_annots['camera']:
            camera_annots = gt_seq_annots['camera'][str(img_id)]
        else:
            camera_annots = gt_seq_annots['camera']["{:06}".format(img_id)]
        
        # load ground truth annotation related info, e.g., bbox, bbox_visib
        if str(img_id) in gt_seq_annots['gt_info']:
            gt_infos = gt_seq_annots['gt_info'][str(img_id)]
        else:
            gt_infos = gt_seq_annots['gt_info']["{:06}".format(img_id)]
        
        
        gt_obj_num = len(gt_pose_annots)
        gt_rotations, gt_translations, gt_labels, gt_bboxes = [], [], [], []
        gt_mask_paths = []
        for i in range(gt_obj_num):
            obj_id = gt_pose_annots[i]['obj_id']
            if self.target_label is not None:
                if obj_id not in self.target_label:
                    continue
            if self.label_mapping is not None:
                if obj_id not in self.label_mapping:
                    continue
                obj_id = self.label_mapping[obj_id]
            px_count_visib = gt_infos[i]['px_count_visib']
            if px_count_visib == 0:
                continue
            gt_labels.append(obj_id)
            gt_rotations.append(np.array(gt_pose_annots[i]['cam_R_m2c'], dtype=np.float32).reshape(3, 3))
            gt_translations.append(np.array(gt_pose_annots[i]['cam_t_m2c'], dtype=np.float32).reshape(-1))
            gt_bboxes.append(np.array(gt_infos[i]['bbox_obj'], dtype=np.float32).reshape(-1))
            mask_path = osp.join(self.gt_annots_root, self.mask_path_tmpl.format(seq_name, img_id, i))
            gt_mask_paths.append(mask_path)

        if len(gt_rotations) == 0:
            raise RuntimeError(f"{img_path} found no gt")
        gt_rotations = np.stack(gt_rotations, axis=0)
        gt_translations = np.stack(gt_translations, axis=0)
        gt_labels = np.array(gt_labels, dtype=np.int64)
        gt_bboxes = np.stack(gt_bboxes, axis=0)
        # ground truth bboxes are xywh format
        gt_bboxes[..., 2:] = gt_bboxes[..., :2] + gt_bboxes[..., 2:]
        gt_obj_num = len(gt_rotations)
        
        formatted_gt_rotations, formatted_gt_translations, formatted_gt_bboxes, formatted_gt_mask_paths = [], [], [], []
        ref_obj_num = len(ref_pose_annots)
        if ref_obj_num > 0:
            ref_rotations, ref_translations, ref_labels = [], [], []
            for i in range(ref_obj_num):
                obj_id = ref_pose_annots[i]['obj_id']
                if self.target_label is not None:
                    if obj_id not in self.target_label:
                        continue
                if self.label_mapping is not None:
                    if obj_id not in self.label_mapping:
                        continue
                    obj_id = self.label_mapping[obj_id]
                translation = np.array(ref_pose_annots[i]['cam_t_m2c'], dtype=np.float32).reshape(-1)
                if self.filter_invalid_pose:
                    if translation[-1] > self.depth_range[-1] or translation[-1] < self.depth_range[0]:
                        continue
                if obj_id not in gt_labels:
                    continue
                ref_rotations.append(np.array(ref_pose_annots[i]['cam_R_m2c'], dtype=np.float32).reshape(3, 3))
                ref_translations.append(translation)
                ref_labels.append(obj_id)

                gt_index = np.nonzero(gt_labels == obj_id)[0][0]
                formatted_gt_rotations.append(gt_rotations[gt_index])
                formatted_gt_translations.append(gt_translations[gt_index])
                formatted_gt_bboxes.append(gt_bboxes[gt_index])
                formatted_gt_mask_paths.append(gt_mask_paths[gt_index])
        else:
            ref_rotations = np.zeros((0, 3, 3), dtype=np.float32)
            ref_translations = np.zeros((0, 3), dtype=np.float32)
            ref_keypoints_3d = np.zeros((0, 8, 3), dtype=np.float32)
        
        ref_translations = np.stack(ref_translations, axis=0)
        ref_rotations = np.stack(ref_rotations, axis=0)
        ref_labels = np.array(ref_labels, dtype=np.int64) - 1
        keypoints_3d = self.keypoints_3d[ref_labels]
        formatted_gt_rotations = np.stack(formatted_gt_rotations, axis=0)
        formatted_gt_translations = np.stack(formatted_gt_translations, axis=0)
        formatted_gt_bboxes = np.stack(formatted_gt_bboxes, axis=0)
        k_orig = np.array(camera_annots['cam_K'], dtype=np.float32).reshape(3,3)
        k = np.repeat(k_orig[None], repeats=ref_translations.shape[0], axis=0)
        

        results_dict = dict()
        results_dict['pose_fields'] = [('gt_rotations', 'gt_translations', 'gt_keypoints_3d'), ('ref_rotations', 'ref_translations', 'ref_keypoints_3d')]
        results_dict['bbox_fields'] = ['gt_bboxes', 'ref_bboxes']
        results_dict['label_fields'] = ['labels']
        results_dict['mask_fields'] = []
        results_dict['annot_fields'] = results_dict['bbox_fields'] + results_dict['label_fields'] + results_dict['mask_fields']\
                                        + list(itertools.chain(*results_dict['pose_fields'])) + ['k', 'ori_k', 'transform_matrix']
        results_dict['aux_fields'] = results_dict['bbox_fields'] + results_dict['label_fields']
        results_dict['ref_rotations'] = ref_rotations
        results_dict['ref_translations'] = ref_translations
        results_dict['gt_rotations'] = formatted_gt_rotations
        results_dict['gt_translations'] = formatted_gt_translations
        results_dict['ref_keypoints_3d'] = keypoints_3d
        results_dict['gt_keypoints_3d'] = keypoints_3d
        results_dict['keypoints_3d'] = keypoints_3d
        results_dict['labels'] = ref_labels
        results_dict['gt_bboxes'] = formatted_gt_bboxes
        results_dict['k'] = k
        results_dict['ori_k'] = k_orig
        results_dict['img_path'] = img_path
        results_dict['gt_mask_path'] = gt_mask_paths
        results_dict['ori_gt_rotations'] = formatted_gt_rotations.copy()
        results_dict['ori_gt_translations'] = formatted_gt_translations.copy()
        results_dict['ori_ref_rotations'] = ref_rotations.copy()
        results_dict['ori_ref_translations'] = ref_translations.copy()
        results_dict = self.transformer(results_dict)
        if results_dict is None:
            raise RuntimeError(f"Data pipeline is broken for image {img_path}")

        return results_dict