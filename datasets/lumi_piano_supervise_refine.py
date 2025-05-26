import itertools
import mmcv
import mmengine
import numpy as np
import random
from tqdm import tqdm
from os import path as osp
from typing import Sequence, Optional
from registry import DATASETS
from torch.utils.data import Dataset
from terminaltables import AsciiTable
import glob
import trimesh

from .pipelines import Compose

@DATASETS.register_module()
class LUMIPianoSuperviseTrainDataset(Dataset):
    def __init__(
        self,
        data_root,
        track_start,
        track_end,
        pipeline,
        keypoints_json: str,
        keypoints_num: int,
        class_names: tuple,
        sample_num=1,
        min_visib_fract=0.,
        min_visib_px_num=0,
        track_prefix=None,
        num_digit=2,
        annot_prefix=None, 
        label_mapping: dict = None,
        target_label: list = None,
        meshes_eval: str = None,
        mesh_symmetry_types: dict = {},
        mesh_diameter: list = [],
        metainfo=None
    ):
        super().__init__()
        self.metainfo = metainfo
        self.data_root = data_root
        self.keypoints_num = keypoints_num
        self.class_names = class_names
        self.label_mapping = label_mapping
        self.target_label = target_label
        self.track_prefix = "" if track_prefix is None else track_prefix
        self.num_digit = num_digit
        self.annot_prefix = "" if annot_prefix is None else annot_prefix
        self.mesh_symmetry_types = mesh_symmetry_types
        self.mesh_diameter = np.array(mesh_diameter)
        self.track_start = track_start
        self.track_end = track_end
        self.min_visib_fract = min_visib_fract
        self.min_visib_px_num = min_visib_px_num
        self.sample_num = sample_num
        if meshes_eval is not None:
            self.meshes = self._load_mesh(meshes_eval, ext='.obj')
        else:
            self.meshes = None
        
        if pipeline is not None:
            self.transformer = Compose(pipeline)

        self.keypoints_3d = self._load_keypoints_3d(keypoints_json)
        self.mask_path_tmpl = "data/{}/mask_visib/{:05}_{:05}.png"
        self.gt_seq_pose_annots, self.img_files = self._load_data()
        if self.label_mapping is not None:
            self.inverse_label_mapping = {v:k for k, v in self.label_mapping.items()}
        else:
            self.inverse_label_mapping = {i+1:i+1 for i in range(len(self.class_names))}

        self.cal_total_sample_num()

    def full_init(sefl):
        pass

    def cal_total_sample_num(self):
        table_data = [['class'] + list(self.class_names) + ['total']]
        sample_num_per_obj = {name:{'total_sample_num':0, 'valid_sample_num':0} for name in self.class_names}
        for track_num in self.gt_seq_pose_annots:
            gt_seq_infos, gt_seq_pose_annots = self.gt_seq_pose_annots[track_num]['gt_info'], self.gt_seq_pose_annots[track_num]['pose']
            for k in gt_seq_infos:
                gt_img_infos = gt_seq_infos[k]
                gt_img_pose_annots = gt_seq_pose_annots[k]
                for i in range(len(gt_img_infos)):
                    obj_info, obj_annot = gt_img_infos[i], gt_img_pose_annots[i]
                    ori_label = obj_annot['obj_id']
                    sample_num_per_obj[self.class_names[ori_label-1]]['total_sample_num'] += 1
                    if self.label_mapping is not None:
                        if ori_label not in self.label_mapping:
                            continue
                        label = self.label_mapping[ori_label]
                    else:
                        label = ori_label
                    if self.target_label is not None:
                        if label not in self.target_label:
                            continue
                    if obj_info['visib_fract'] < self.min_visib_fract:
                        continue
                    sample_num_per_obj[self.class_names[ori_label-1]]['valid_sample_num'] += 1
    
        for k in ['total_sample_num', 'valid_sample_num']:
            table_data.append(
                [k] + [sample_num_per_obj[name][k] for name in sample_num_per_obj] + [sum([sample_num_per_obj[name][k] for name in sample_num_per_obj])]
            )
        self.total_sample_num = AsciiTable(table_data).table

    def _load_data(self):
        ann_files=[
            osp.join(self.data_root, f"data/{self.track_prefix}{''.join(['0'] * (self.num_digit-1)) + str(t)}/image_set/{self.annot_prefix}_train.txt")
            for t in range(self.track_start, self.track_end + 1) 
        ]
        image_prefixes=[
            osp.join(self.data_root, f"data/{self.track_prefix}{''.join(['0'] * (self.num_digit-1)) + str(t)}")
            for t in range(self.track_start, self.track_end + 1)
        ]
    
        assert len(ann_files) == len(image_prefixes)
        img_files = []
        gt_seq_pose_annots = dict()

        for ann_file, scene_root in zip(ann_files, image_prefixes):
            with open(str(ann_file), 'r') as f_ann:
                indices = [line.strip("\r\n") for line in f_ann.readlines()]  # string ids

            for im_id in tqdm(indices):
                int_im_id = int(im_id)
                rgb_path = osp.join(scene_root, "rgb/{:05d}.png").format(int_im_id)
                assert osp.exists(rgb_path), rgb_path
                img_files.append(rgb_path) 

            gt_pose_json_path = osp.join(scene_root, "scene_gt.json")
            gt_info_json_path = osp.join(scene_root, "scene_gt_info.json")  # bbox_obj, bbox_visib
            camera_json_path = osp.join(scene_root, "scene_camera.json")

            gt_pose_annots = mmengine.load(gt_pose_json_path)
            gt_infos = mmengine.load(gt_info_json_path)
            camera_annots = mmengine.load(camera_json_path)

            scene = scene_root.split("/")[-1]
            gt_seq_pose_annots[scene] = dict(
                pose=gt_pose_annots, 
                camera=camera_annots, 
                gt_info=gt_infos
            )
        
        return gt_seq_pose_annots, img_files

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
    
    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += f'image_num={len(self)}, '
        s += f"sample num info: \n {self.total_sample_num} \n"
        return s

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        results = self.getitem(index)
        while results is None:
            index = random.randint(0, len(self.img_files) - 1)
            results = self.getitem(index)
        return results
    
    def getitem(self, index):
        img_path = self.img_files[index]
        _,  seq_name, _, img_name = img_path.rsplit('/', 3)
        img_id = int(osp.splitext(img_name)[0])
        
        gt_seq_annots = self.gt_seq_pose_annots[seq_name]
        # load ground truth pose annots
        if str(img_id) in gt_seq_annots['pose']:
            gt_pose_annots = gt_seq_annots['pose'][str(img_id)]
        else:
            gt_pose_annots = gt_seq_annots['pose']["{:06}".format(img_id)]
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
            visib_fract = gt_infos[i]['visib_fract']
            if visib_fract < self.min_visib_fract:
                continue
            visib_px_count = gt_infos[i]['px_count_visib']
            if visib_px_count < self.min_visib_px_num:
                continue
            gt_rotations.append(np.array(gt_pose_annots[i]['cam_R_m2c'], dtype=np.float32).reshape(3, 3))
            gt_translations.append(np.array(gt_pose_annots[i]['cam_t_m2c'], dtype=np.float32).reshape(-1))
            gt_labels.append(obj_id)
            gt_bboxes.append(np.array(gt_infos[i]['bbox_obj'], dtype=np.float32).reshape(-1))
            if 'mask_id' in gt_infos[i]:
                mask_path = osp.join(self.data_root, self.mask_path_tmpl.format(img_id, gt_infos[i]['mask_id']))
            else:
                mask_path = osp.join(self.data_root, self.mask_path_tmpl.format(seq_name, img_id, i))
            gt_mask_paths.append(mask_path)

        if len(gt_labels) == 0:
            return None
        gt_rotations = np.stack(gt_rotations, axis=0)
        gt_translations = np.stack(gt_translations, axis=0)
        gt_labels = np.array(gt_labels, dtype=np.int64) - 1
        gt_keypoints_3d = self.keypoints_3d[gt_labels]
        gt_bboxes = np.stack(gt_bboxes, axis=0)
        # ground truth bboxes are xywh format
        gt_bboxes[..., 2:] = gt_bboxes[..., :2] + gt_bboxes[..., 2:]
        obj_num = len(gt_rotations)

        if self.sample_num == -1:
            sample_num = obj_num
        else:
            sample_num = self.sample_num
        choosen_obj_index = np.random.choice(list(range(obj_num)), sample_num, replace=False)
        gt_rotations = gt_rotations[choosen_obj_index]
        gt_translations = gt_translations[choosen_obj_index]
        gt_labels = gt_labels[choosen_obj_index]
        gt_bboxes = gt_bboxes[choosen_obj_index]
        gt_keypoints_3d = gt_keypoints_3d[choosen_obj_index]
        gt_mask_paths = np.array(gt_mask_paths)[choosen_obj_index].tolist()
        
        k = np.array(camera_annots['cam_K'], dtype=np.float32).reshape(3,3)
        k = np.repeat(k[None], repeats=sample_num, axis=0)
        results_dict = dict()

        results_dict['pose_fields'] = [('gt_rotations', 'gt_translations', 'gt_keypoints_3d'), ('ref_rotations', 'ref_translations', 'ref_keypoints_3d')]
        results_dict['bbox_fields'] = ['gt_bboxes', 'ref_bboxes']
        results_dict['mask_fields'] = ['gt_masks']
        results_dict['label_fields'] = ['labels']
        
        results_dict['annot_fields'] = results_dict['bbox_fields'] + results_dict['mask_fields'] + results_dict['label_fields'] \
                                        + list(itertools.chain(*results_dict['pose_fields'])) + ['k']
        results_dict['aux_fields'] = results_dict['bbox_fields'] + results_dict['mask_fields'] + results_dict['label_fields']
        results_dict['gt_rotations'] = gt_rotations
        results_dict['gt_translations'] = gt_translations
        results_dict['gt_keypoints_3d'] = gt_keypoints_3d
        results_dict['ref_keypoints_3d'] = gt_keypoints_3d
        results_dict['ori_gt_rotations'] = gt_rotations.copy()
        results_dict['ori_gt_translations'] = gt_translations.copy()
        results_dict['labels'] = gt_labels
        results_dict['gt_bboxes'] = gt_bboxes
        results_dict['gt_mask_path'] = gt_mask_paths
        results_dict['k'] = results_dict['ori_k'] = k
        results_dict['img_path'] = img_path
        results_dict = self.transformer(results_dict)

        return results_dict