from typing import Sequence, List, Any, Optional
import numpy as np
import random
from mmengine.evaluator import BaseMetric
from terminaltables import AsciiTable
from pathlib import Path
from mmengine.logging import print_log
import mmengine
import os.path as osp
from tqdm import tqdm
import glob
import trimesh
import itertools
from datasets import Compose
from datasets.utils import dumps_json

from registry import METRICS
from datasets.pose import project_3d_point
from .add import ADD

@METRICS.register_module()
class LUMIPianoADD(ADD):

    def __init__(self,
                data_root: str,
                keypoints_json: str,
                class_names: tuple,
                ref_annots_root:str,
                track_start,
                track_end,
                gt_annots_root:str=None,
                target_label: list = None,
                label_mapping: dict = None,
                keypoints_num: int = 8,
                filter_invalid_pose:bool=False,
                depth_range: Optional[tuple]=None,
                meshes_eval: str = None,
                save_dir: str = None,
                mesh_symmetry: dict = {},
                mesh_diameter: list = [],
                metrics: dict = {'auc':[], 'add':[0.05, 0.10, 0.20, 0.50]}):
        super(ADD, self).__init__()
        self.data_root = data_root
        self.keypoints_num = keypoints_num
        self.class_names = class_names
        self.label_mapping = label_mapping
        self.target_label = target_label
        self.mesh_symmetry_types = mesh_symmetry
        self.mesh_diameter = np.array(mesh_diameter)
        self.save_dir = save_dir
        self.metrics = metrics
        self.supported_metric = ['add', 'rep']
        self.track_start = track_start
        self.track_end = track_end
        if meshes_eval is not None:
            self.meshes = self._load_mesh(meshes_eval, ext='.obj')
        else:
            self.meshes = None
        
        self.keypoints_3d = self._load_keypoints_3d(keypoints_json)
        if self.label_mapping is not None:
            self.inverse_label_mapping = {v:k for k, v in self.label_mapping.items()}
        else:
            self.inverse_label_mapping = {i+1:i+1 for i in range(len(self.class_names))}
        
        if gt_annots_root is not None:
            self.gt_annots_root = gt_annots_root
            self.gt_seq_pose_annots = self._load_pose_annots()

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

    # TODO: Imp.
    def process(self, data_batch, data_samples):
        """Process one batch of data and predictions. The processed
        Results should be stored in `self.results`, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        self.results.extend(data_samples)

    def compute_metrics(self, results):
        '''
        Evaluation.
        Args:
            results (list[dict]): Testing results of the whole test dataset.
                Each entry is formated as following:
                    dict(
                        "gt": {'labels':[], 'translations':[], 'rotations':[],},
                        "pred": {'labels':[], 'scores':[], 'translations':[], 'rotations':[],},
                        "img_metas": {},
                    ),
                The keys annotated by "*" may not exist.
            metric (dict): Metrics to be evaluated. Support Add and Rep(2d).
            logger : Logger used for printing related information during evvaluation.
        '''
        metrics = self.metrics.copy()

        self.format_results(results, self.save_dir)
        
        rotation_gts, translation_gts, rotation_preds, translation_preds, labels, matched_gt_flag, concat_k = self.match_results(results)
        error_dict = {}
        if 'add' in metrics or 'rep' in metrics:
            vertices_list = [mesh.vertices.view(np.ndarray).astype(np.float32) for mesh in self.meshes]
            vertices_list = [vertices[np.random.choice(vertices.shape[0], 1000)] for vertices in vertices_list]
            error_3d_normalized = np.ones_like(labels, dtype=np.float32)
            error_3d = np.full_like(labels, dtype=np.float32, fill_value=110)
            error_2d = np.full_like(labels, dtype=np.float32, fill_value=50.)
            error_3d_normalized_, error_2d_, error_3d_ = self.eval_pose_error(vertices_list,
                                                          gt_t=translation_gts[matched_gt_flag],
                                                          gt_r=rotation_gts[matched_gt_flag],
                                                          pred_t=translation_preds[matched_gt_flag],
                                                          pred_r=rotation_preds[matched_gt_flag],
                                                          labels=labels[matched_gt_flag],
                                                          k=concat_k[matched_gt_flag],
                                                          symmetry_types=self.mesh_symmetry_types,
                                                          mesh_diameters=self.mesh_diameter)
            error_3d_normalized[matched_gt_flag] = error_3d_normalized_
            error_2d[matched_gt_flag] = error_2d_
            error_3d[matched_gt_flag] = error_3d_
            error_dict['add'] = error_3d_normalized
            error_dict['rep'] = error_2d

        metric_dict, headers = self.parse_error_to_metric(
            error_dict=error_dict,
            labels=labels,
            metrics=metrics,
            classnames=self.class_names,
        )
        self.print_metric(metric_dict, headers, logger=None)
        return self.parse_metric_to_tensorboard(metric_dict, headers)

    def match_results(self, results):
        formatted_gt_rotations, formatted_gt_translations = [], []
        formatted_pred_rotations, formatted_pred_translations = [], []
        valid_flag, labels = [], []
        formatted_k = []
        vertices_list = [mesh.vertices.view(np.ndarray).astype(np.float32) for mesh in self.meshes]
        vertices_list = [vertices[np.random.choice(vertices.shape[0], 1000)] for vertices in vertices_list]
        print('formatting results....')
        pbar = tqdm(results)
        for result in pbar:
            image_path = result['img_metas']['img_path']
            if len(image_path.rsplit('/', 3)) == 3:
                seq_name, _, img_name = image_path.rsplit('/', 3)
            else: 
                _, seq_name, _, img_name = image_path.rsplit('/', 3)
            img_id = int(osp.splitext(img_name)[0])
            gt_seq_annots = self.gt_seq_pose_annots[seq_name]
            gt_pose_annots = gt_seq_annots['pose'][str(img_id)]
            camera_annots = gt_seq_annots['camera'][str(img_id)]
            pred = result['pred']
            pred_labels, pred_rotations, pred_translations = pred['labels'], pred['rotations'], pred['translations']
            k = np.array(camera_annots['cam_K'], dtype=np.float32).reshape(3,3)
            for i, l in enumerate(pred_labels):
                pred_labels[i] = self.inverse_label_mapping[l + 1]
            
            gt_obj_num = len(gt_pose_annots)
            for i in range(gt_obj_num):
                obj_id = gt_pose_annots[i]['obj_id']
                matched_preds = pred_labels == obj_id
                matched_pred_num = matched_preds.sum()
                gt_rotation = np.array(gt_pose_annots[i]['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
                gt_translation = np.array(gt_pose_annots[i]['cam_t_m2c'], dtype=np.float32).reshape(-1)
                formatted_gt_translations.append(gt_translation)
                formatted_gt_rotations.append(gt_rotation)
                formatted_k.append(k)
                labels.append(obj_id)
                if matched_pred_num == 1:
                    matched_index = np.nonzero(matched_preds)[0][0]
                    formatted_pred_rotations.append(pred_rotations[matched_index])
                    formatted_pred_translations.append(pred_translations[matched_index])
                    valid_flag.append(True)
                elif matched_pred_num > 1:
                    error_3d_normalized_, _, _ = self.eval_pose_error(
                        vertices_list, 
                        gt_r=np.repeat(gt_rotation[None], matched_pred_num, 0),
                        gt_t=np.repeat(gt_translation[None], matched_pred_num, 0),
                        pred_r=pred_rotations[matched_preds],
                        pred_t=pred_translations[matched_preds],
                        labels=np.repeat(obj_id, matched_pred_num, 0)-1,
                        k=np.repeat(k[None], matched_pred_num, 0),
                        symmetry_types=self.mesh_symmetry_types,
                        mesh_diameters=self.mesh_diameter,
                    )
                    error_3d_normalized = np.full_like(matched_preds, fill_value=100, dtype=np.float32)
                    error_3d_normalized[matched_preds] = error_3d_normalized_
                    matched_index = np.argmin(error_3d_normalized)
                    
                    formatted_pred_rotations.append(pred_rotations[matched_index])
                    formatted_pred_translations.append(pred_translations[matched_index]) 
                    valid_flag.append(True)
                else:
                    formatted_pred_rotations.append(np.zeros((3, 3), dtype=np.float32))
                    formatted_pred_translations.append(np.zeros((3,), dtype=np.float32)) 
                    valid_flag.append(False)

        formatted_gt_rotations = np.stack(formatted_gt_rotations, axis=0)
        formatted_gt_translations = np.stack(formatted_gt_translations, axis=0)
        formatted_pred_rotations = np.stack(formatted_pred_rotations, axis=0)
        formatted_pred_translations = np.stack(formatted_pred_translations, axis=0)
        formatted_k = np.stack(formatted_k, axis=0)
        valid_flag = np.array(valid_flag, dtype=np.bool_)
        labels = np.array(labels, dtype=np.int64)
        return formatted_gt_rotations, formatted_gt_translations, formatted_pred_rotations, formatted_pred_translations, labels-1, valid_flag, formatted_k



    def parse_error_to_metric(self, error_dict, labels, metrics, classnames=None):
        '''
        Args:
            error_list (dict[np.ndarray]): Different kinds of error, like mask error, pose error, etc.
            labels (np.ndarray): Labels 
            metrics (dict(list)): Different kind of metrics.
            classnames (list | tuple): The elemet is str, stands for the category name.
            classwise (bool): Perform classwise calcuclation.
        '''
        metric_dict = {'average':[]}
        headers = ['class']
        average_precision_total = []
        classwise_precision = {classname:[] for classname in classnames}

        for metric in metrics:
            if metric not in self.supported_metric:
                continue
            error = error_dict[metric]
            thresholds = metrics[metric]
            if len(thresholds) == 0:
                # already precision, like 'epe', 'similarity'
                headers.append(metric)
                valid_class_num = 0
                if metric == 'auc':
                    average_precision = []
                    for l in range(len(classnames)):
                        if (labels == l).sum() == 0:
                            classwise_precision[classnames[l]].append(-1)
                        else:
                            auc_per_category = self.eval_auc_metric(error[labels==l], max_error=100)
                            classwise_precision[classnames[l]].append(auc_per_category)
                            average_precision.append(auc_per_category)
                    average_precision_total.append(average_precision)
                else:
                    for l in range(len(classnames)):
                        if (labels == l).sum() == 0:
                            classwise_precision[classnames[l]].append(-1)
                        else:
                            classwise_precision[classnames[l]].append(error[labels==l].mean())
                            valid_class_num += 1
                    average_precision_total.append(error.tolist())    
            else:
                # update headers
                for thr in thresholds:
                    if thr < 1:
                        headers.append("{}_{:0>2d}".format(metric ,int(thr*100)))
                    else:
                        headers.append("{}_{:0>2d}".format(metric ,thr))

                
                # calculate classwise precision
                average_precision = [[] for _ in range(len(thresholds))]
                for l in range(len(classnames)):
                    error_label = error[labels == l]
                    # If can't find this category, leave the precision as -1
                    if error_label.shape[0] == 0:
                        label_precision = [-1.0] * len(thresholds)
                    else:
                        label_precision = []
                        for i, thr in enumerate(thresholds):
                            label_precision_per_thr = (error_label < thr).sum() / error_label.shape[0]
                            label_precision.append(label_precision_per_thr)
                            average_precision[i].append(label_precision_per_thr)
                    classwise_precision[classnames[l]].extend(label_precision)
                average_precision_total.extend(average_precision)
    
        metric_dict.update(classwise_precision)
        # average the classwise precision to average precision
        average_precision = [sum(precision)/len(precision) for precision in average_precision_total]
        metric_dict['average'] = average_precision
        return metric_dict, headers
    
    
    def print_metric(self, metric_dict, headers, logger):
        table_data = [headers]
        for class_name in metric_dict:
            class_msg = [class_name]
            metric_dict[class_name] = list(map(lambda x:round(x, 4), metric_dict[class_name]))
            precision = metric_dict[class_name]
            table_data.append(class_msg + list(map(lambda x:round(x, 4), precision)))
        table = AsciiTable(table_data)
        print_log('\n' + table.table, logger)
    
    def parse_metric_to_tensorboard(self, metric_dict, headers):
        new_metric_dict = {}
        for class_name in metric_dict:
            for i, metric_type in enumerate(headers):
                if metric_type == 'class':
                    continue
                new_metric_dict[f'{class_name}/{metric_type}'] = metric_dict[class_name][i-1]
        return new_metric_dict

    
    def eval_pose_error(self, verts_list, gt_t, gt_r, pred_t, pred_r, 
                            labels, k, symmetry_types, mesh_diameters):
        '''
        Args:
            verts_list (list[np.ndarray]): verts for each object(class) model.
            gt_t (np.ndarray): Ground truth translations, shape (N, 3, 1)
            gt_r (np.ndarray): Ground truth rotations, shape (N, 3, 3)
            pred_t (np.ndarray): Predicted translations, shape (N, 3, 1)
            pred_r (np.ndarray): Predicted rotations, shape (N, 3, 3)
            labels (np.ndarray): Ground truth labels, shape (N)
            k (np.ndarray): Camera intrinsic parameters, shape (N, 3, 3)
            symmetry_types (dict): dataset symmetry types, 
                If mesh is symmetric, will use add-s metric.
            mesh_diameteres (list[float]): Normalize pose errors.

        '''
        # TODO: Seperate 3d error calculate and 2d error calculate
        
        num_pred = len(gt_t)
        error_3d_normalized, error_2d, error_3d = np.zeros((num_pred)), np.zeros((num_pred)), np.zeros((num_pred))
        total_labels = np.unique(labels)
        
        for i in total_labels:
            cls_i_index = labels == i
            verts = verts_list[i]
            gt_t_cls_i, gt_r_cls_i = gt_t[cls_i_index], gt_r[cls_i_index]
            pred_t_cls_i, pred_r_cls_i = pred_t[cls_i_index], pred_r[cls_i_index]
            cls_i_k = k[cls_i_index]

            gt_2d, gt_3d = project_3d_point(verts, cls_i_k, gt_r_cls_i, gt_t_cls_i[..., None], return_3d=True)
            pred_2d, pred_3d = project_3d_point(verts, cls_i_k, pred_r_cls_i, pred_t_cls_i[..., None], return_3d=True)

            if symmetry_types.get(f"cls_{i+1}", False):
                pred_3d_list = []
                for gt_3d_per_obj, pred_3d_per_obj in zip(gt_3d, pred_3d):
                    ext_gt = np.expand_dims(gt_3d_per_obj, -2)
                    ext_pred = np.expand_dims(pred_3d_per_obj, -3)
                    min_idx = np.argmin(np.linalg.norm(ext_gt - ext_pred, axis=-1), axis=-1)
                    pred_3d_per_obj = pred_3d_per_obj[min_idx]
                    pred_3d_list.append(pred_3d_per_obj)
                pred_3d = np.stack(pred_3d_list, axis=0)
            
            error = (np.linalg.norm(gt_3d - pred_3d, axis=-1).mean(axis=-1))
            error_3d_normalized[cls_i_index] = error/mesh_diameters[i]
            error_2d[cls_i_index] = np.linalg.norm(gt_2d - pred_2d, axis=-1).mean(axis=-1)
            error_3d[cls_i_index] = error
        return error_3d_normalized, error_2d, error_3d

    def format_results(self, results, save_dir, time=None):
        '''
        Args:
            results (list[dict]): Testing results of the dataset
            save_dir (str): The directory to save the results in BOP format.
            match_results (bool): Only save the results matching the target category.
        
        '''
        if save_dir is None:
            return
        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True)
        sequence_gts = dict()
        for i, result in enumerate(results):
            src_path = result['img_metas']['img_path']
            dst_path = src_path.replace(self.data_root, save_dir)
            sequence_path = Path(dst_path).parents[1]
            if not sequence_path.exists():
                sequence_path.mkdir(parents=True)
            sequence_path = str(sequence_path)
            if sequence_path not in sequence_gts:
                sequence_gts[sequence_path] = dict()
            id = str(int(Path(dst_path).stem))
            assert id not in sequence_gts[sequence_path]
            # predictions
            preds_orig = result['pred']
            translations = preds_orig['translations']
            rotations = preds_orig['rotations']
            obj_ids = (preds_orig['labels'] + 1).tolist()
            
        
            preds = []
            num_preds = len(translations)
            for i in range(num_preds):
                obj_id = self.inverse_label_mapping[obj_ids[i]]
                res = dict(
                    cam_R_m2c=rotations[i].reshape(-1).tolist(),
                    cam_t_m2c= translations[i].tolist(),
                    obj_id=obj_id,
                )
                if time is not None:
                    res.update(time=time)
                preds.append(res)
            
            sequence_gts[sequence_path][id] = preds

        for sequence in sequence_gts:
            # save pose preds
            save_path = Path(sequence).joinpath('scene_gt.json')
            save_content = sequence_gts[sequence]
            dump_content = dumps_json(save_content)
            save_path.write_text(dump_content) 
