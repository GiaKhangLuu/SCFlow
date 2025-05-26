from typing import Optional, Sequence, Tuple
import mmcv
import mmengine
import numpy as np
import pyrender
import cv2
import torch
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from datasets.pose import project_3d_point
from models.utils.rendering import Renderer

EPS = 1e-2

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def draw_3d_bbox(img, imgpts, color, size=3):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, size)

    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, size)

    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, size)
    return img

def draw_3d_pts(img, imgpts, color, size=1):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    for point in imgpts:
        img = cv2.circle(img, (point[0], point[1]), size, color, -1)
    return img

def draw_detections(image, pred_rots, pred_trans, model_points, intrinsics, color=(255, 0, 0)):
    num_pred_instances = len(pred_rots)
    draw_image_bbox = image.copy()
    # 3d bbox
    scale = (np.max(model_points, axis=0) - np.min(model_points, axis=0))
    shift = np.mean(model_points, axis=0)
    bbox_3d = get_3d_bbox(scale, shift)

    # 3d point
    choose = np.random.choice(np.arange(len(model_points)), 512)
    pts_3d = model_points[choose].T

    for ind in range(num_pred_instances):
        # draw 3d bounding box
        transformed_bbox_3d = pred_rots[ind]@bbox_3d + pred_trans[ind][:,np.newaxis]
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics[ind])
        draw_image_bbox = draw_3d_bbox(draw_image_bbox, projected_bbox, color)
        # draw point cloud
        transformed_pts_3d = pred_rots[ind]@pts_3d + pred_trans[ind][:,np.newaxis]
        projected_pts = calculate_2d_projections(transformed_pts_3d, intrinsics[ind])
        draw_image_bbox = draw_3d_pts(draw_image_bbox, projected_pts, color)

    return draw_image_bbox

def color_val(color):
    if mmengine.utils.is_str(color):
        # see https://matplotlib.org/stable/gallery/color/named_colors.html for full list of color names
        return tuple(map(lambda x:int(x*255), colors.to_rgba(color)[:3]))[::-1]
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError(f'Invalid type for color: {type(color)}')
    



def color_val_matplotlib(color, RGB=True, normalize=True):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = color_val(color)
    if normalize:
        color = [color / 255 for color in color[::-1]]
    else:
        color = [color for color in color[::-1]]
    if not RGB:
        color.reverse()
    return tuple(color)




def imshow_projected_points(img,
                            object_meshes,
                            rotations,
                            translations,
                            labels,
                            camera_k,
                            scores=None,
                            show_points_num=1000,
                            class_names=None,
                            score_thr=0,
                            point_color='green',
                            point_size=1,
                            win_name='',
                            wait_time=0,
                            show=True,
                            out_file=None):
    '''
    Visualize projected mesh points, determined by rotations, translations and camera_k.
    Args:
        img (str or np.ndarray): The image to de displayed.
        obejct_meshses (list[np.ndarray]): Object 3d meshes.
        rotations (np.ndarray): Rotations, shape (n, 3, 3)
        translations (np.ndarray): Transaltions, shape (n, 3, 1)
        labels (np.ndarray): Labels, shape (n)
        camera_k (np.ndarray): Camera intrinsic, shape (3, 3) or (n, 3, 3)
        scores (np.ndarray): Scores of each predicted element.
        show_points_num (int): Randomly choose points from mesh to visualize.
        class_names (list[str]): Names of each category.
        score_thr (float): Filter predictions by score threshold.
        point_color (str or tuple(int) or list[str] or list[tuple(int)]): Color of points.
            Either a list for each class or a single element for all classes
        win_name (str): 
        show (bool):
        out_file (str): The filename to write thr visualized image.
    '''
    if rotations.size == 0:
        return img
    img = mmcv.imread(img).astype(np.uint8)
    if score_thr > 0:
        assert scores is not None
        assert scores.shape[0] == rotations.shape[0] == translations.shape[0]
        inds = scores > score_thr
        rotations = rotations[inds]
        translations = translations[inds]
        labels = labels[inds]
        if camera_k.ndim > 3:
            camera_k = camera_k[inds]
    
    if isinstance(point_color, list):
        point_colors = [color_val_matplotlib(c) for c in point_color]
    else:
        if class_names is not None:
            point_colors = [color_val_matplotlib(point_color)] * (len(class_names) + 1)
        else:
            point_colors = [color_val_matplotlib(point_color)] * (max(labels) + 1)
    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    fig.set_size_inches((width + EPS)/dpi, (height + EPS)/dpi)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    sampled_meshes = [
        mesh[np.random.choice(mesh.shape[0], show_points_num)] 
        for mesh in object_meshes]
        
    plt.imshow(img)
    for i in range(len(rotations)):
        rotation, translation, label = rotations[i], translations[i], labels[i]
        k = camera_k if camera_k.ndim == 2 else camera_k[i]
        mesh = sampled_meshes[label]
        projected_points = project_3d_point(mesh, k, rotation, translation)
        c = point_colors[label]
        plt.scatter(projected_points[:, 0], 
                    projected_points[:, 1], 
                    s=point_size, 
                    color=c,
                    marker='o')

    # from matplotlib to cv2 
    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype(np.uint8)
    img = mmcv.rgb2bgr(img)

    if show:
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
    
    plt.close()
    return img
        

def imshow_pose_axis(img,
                    radius,
                    rotations, 
                    translations,
                    labels,
                    camera_k,
                    scores=None,
                    class_names=None,
                    score_thr=0,
                    axis_color='green',
                    thickness=3,
                    win_name='',
                    wait_time=0,
                    show=True,
                    out_file=None):
    '''
    Visualize pose axis, determined by rotations, translations, and camera_k
    Args:
        img (str or np.ndarray): The image to be displayed.
        rotations (np.ndarray): Rotations, shape (n, 3, 3)
        translations (np.ndarray): Translations, shape (n, 3)
        labels (np.ndarray): Labels, shape (n)
        radius (list): Mesh diameter of each object
        camera_k (np.ndarray): Camera intinsic, shape (3, 3)
        scores (np.ndarray): Scores of each predicted element.
        class_names (list[str]): Names of each category.
        score_thr (float): Filter predictions by score threshold.
        axis_color (str or tuple(int) or list[str] or list[typle(int)]): Color of pose axis.
            Either a list for each class or a single element for all classes.
        win_name (str):
        wait_time (float):
        show (bool):
        out_file (str):
    '''
    if rotations.size == 0:
        return img
    img = mmcv.imread(img).astype(np.uint8)
    if score_thr > 0:
        assert scores is not None
        assert scores.shape[0] == rotations.shape[0] == translations.shape[0]
        inds = scores > score_thr
        rotations = rotations[inds]
        translations = translations[inds]
        labels = labels[inds]
        if camera_k.ndim > 3:
            camera_k = camera_k[inds]
    
    if isinstance(axis_color, list):
        axis_colors = [color_val_matplotlib(c) for c in axis_color]
    else:
        if class_names is not None:
            axis_colors = [color_val_matplotlib(axis_color)] * (len(class_names) + 1)
        else:
            axis_colors = [color_val_matplotlib(axis_color)] * (max(labels) + 1)
    
    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    fig.set_size_inches((width + EPS)/dpi, (height + EPS)/dpi)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')


    plt.imshow(img)

    for i in range(len(rotations)):
        rotation, translation, label = rotations[i], translations[i], labels[i]
        k = camera_k if camera_k.ndim == 2 else camera_k[i]
        r = radius[label]
        points_3d = [(0, 0, 0), (0, 0, r), (0, r, 0), (r, 0, 0)]
        points_3d = np.array(points_3d).reshape(-1, 3)
        points_2d = project_3d_point(points_3d, k, rotation, translation)
        lines = []
        lines.append(np.stack([points_2d[0], points_2d[1]]))
        lines.append(np.stack([points_2d[0], points_2d[2]]))
        lines.append(np.stack([points_2d[0], points_2d[3]]))
        lines = np.stack(lines)
        line_collection = LineCollection(lines, 
                                        color = ['b', 'g', 'r'],
                                        linewidths=thickness)
        ax.add_collection(line_collection)
    
    # from matplotlib to cv2 
    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype(np.uint8)
    img = mmcv.rgb2bgr(img)

    if show:
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
    
    plt.close()
    return img
    


def imshow_pose_contour(img:np.ndarray,
                        object_meshes,
                        rotations,
                        translations,
                        labels,
                        camera_k,
                        scores=None,
                        class_names=None,
                        score_thr=0,
                        color='green',
                        contour_size=3,
                        win_name='',
                        wait_time=0,
                        show=True,
                        out_file=None):
    
    def render_objects(mesh, rotation, translation, camera_k, target_w, target_h):
        fx, fy, cx, cy = camera_k[0, 0], camera_k[1, 1], camera_k[0, 2], camera_k[1, 2]
        scene = pyrender.Scene(bg_color=np.array([1., 1., 1., 0.]), ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=100000)
        camera_pose = np.eye(4)
        camera_pose[1][1] = -1
        camera_pose[2][2] = -1
        scene.add(camera, pose=camera_pose)

        light = pyrender.SpotLight(color=np.ones(3), intensity=4.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
        scene.add(light, pose=camera_pose)

        mesh = pyrender.Mesh.from_trimesh(mesh)
        H = np.zeros((4, 4))
        H[:3, :3] = rotation
        H[:3, 3] = translation.T 
        H[3, 3] = 1.0
        scene.add(mesh, pose=H)

        r = pyrender.OffscreenRenderer(target_w, target_h)
        flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.RGBA
        color, depth = r.render(scene, flags=flags)
        color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA)
        return color, depth

    if rotations.size == 0:
        return img
    img = mmcv.imread(img).astype(np.uint8)
    if score_thr > 0:
        assert scores is not None
        assert scores.shape[0] == rotations.shape[0] == translations.shape[0]
        inds = scores > score_thr
        rotations = rotations[inds]
        translations = translations[inds]
        labels = labels[inds]
        if camera_k.ndim > 3:
            camera_k = camera_k[inds]
    
    if isinstance(color, list):
        colors = [color_val_matplotlib(c, RGB=False, normalize=False) for c in color]
    else:
        if class_names is not None:
            colors = [color_val_matplotlib(color, RGB=False, normalize=False)] * (len(class_names) + 1)
        else:
            colors = [color_val_matplotlib(color, RGB=False, normalize=False)] * (max(labels) + 1)

    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    for i in range(len(rotations)):
        rotation, translation, label = rotations[i], translations[i], labels[i]
        k = camera_k if camera_k.ndim == 2 else camera_k[i]
        mesh = object_meshes[label]
        _, depth = render_objects(mesh, rotation, translation, k, width, height)
        mask = (depth > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(img, contours, -1, colors[label], contour_size)

    if out_file is not None:
        mmcv.imwrite(img, out_file)
    return img


def imshow_2d_keypoints(img,
                        keypoints_2d,
                        labels,
                        scores=None,
                        class_names=None,
                        score_thr=0,
                        keypoint_color='green',
                        order=None,
                        line_width=3,
                        show=True,
                        wait_time=0,
                        win_name='',
                        out_file=None):
    '''
    Visualize projected 2d keypoints. 
    Args:
        img (str or ndarray): The image to be displayed.
        keypoints_2d (ndarray): Projected 2d keypoints, shape (n, keypoint_num, 2)
        labels (ndarray): Labels of keypoints
        class_names (list[str]): Names of each classes.
        keypoint_color (tuple(int)|str or list[tuple(int)]|list[str]): Color of keypoint circles. 
            The tuple of color should be in BGR channel. 
            If provided list[tuple(int)](list[str]), the length should be equal to the number of classes.
        bbox_color (str or tuple(int)): Color of bbox lines.
            The tuple of color should be in BGR channel.
        order (list(tuple), Optional): The connections of keypoints. 
            Typically, 12 lines define a cube. And the order should have a shape (12, 2)
        line_width (int): Thickness of lines. 
        show (bool): whether to show the image. Default:True
        wait_time (float): Value of waitkey param. Default: 0.
        out_file (str): The filename to write the image. Default:None
    '''
    if keypoints_2d.size == 0:
        return img
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert keypoints_2d.shape[0] == labels.shape[0], \
        'keypoints_2d.shape[0] and labels.shape[0] should have the same length.'
    
    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert scores is not None
        assert scores.shape[0] == labels.shape[0]
        inds = scores > score_thr
        keypoints_2d = keypoints_2d[inds, :]
        labels = labels[inds]
    
    if isinstance(keypoint_color, list):
        keypoint_colors = [color_val_matplotlib(c) for c in keypoint_color]

    else:
        if class_names is not None:
            keypoint_colors = [color_val_matplotlib(keypoint_color)] * (len(class_names) + 1)
        else:
            keypoint_colors = [color_val_matplotlib(keypoint_color)] * (max(labels) + 1)
    

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    fig.set_size_inches((width + EPS)/dpi, (height + EPS)/dpi)

    # remove white edges
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')
    
    lines = []
    colors = []
    for i, keypoint in enumerate(keypoints_2d):
        keypoint_int = keypoint.astype(np.int32).reshape(-1, 1, 2)
        
        if order is not None:
            lines.append([
                np.concatenate([keypoint_int[o[0]], keypoint_int[o[1]]], axis=0) 
                for o in order])
        else:
            lines.append(keypoint_int)
        colors.append(np.tile(
                    np.array(keypoint_colors[labels[i]]).reshape(1, -1), 
                    (keypoint_int.shape[0], 1)))
        for i, k in enumerate(keypoint_int):
            plt.text(k[0, 0], k[0, 1], str(i+1), color='r')
        
    plt.imshow(img)
    lines = np.concatenate(lines, axis=0)
    colors = np.concatenate(colors)
    if lines.shape[1] == 1:
        # draw points
        plt.scatter(lines[:, 0, 0], lines[:, 0, 1],
                    s=line_width, c=colors, marker='o')
    else:
        # draw lines
        line_collection = LineCollection(
            lines, 
            color=colors, 
            linewidths=np.array([line_width]*lines.shape[0]),
        )
        ax.add_collection(line_collection)


    # from matplotlib to cv2 
    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype(np.uint8)
    img = mmcv.rgb2bgr(img)

    if show:
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
    
    plt.close()
    return img



class Pytorch3dVisTool:
    def __init__(self, mesh_dir:str, image_size:Sequence[int], colors:Sequence, score_thr=0., vis_mode:str='mask', contour_size=3):
        self.render = Renderer(
            mesh_dir, image_size,
            soft_blending=False, render_mask=False,
            sigma=1e-12, gamma=1e-12, bin_size=-1)
        self.render.to('cuda')
        self.score_thr = score_thr
        self.colors = [color_val_matplotlib(color, RGB=False, normalize=False) for color in colors]
        assert vis_mode in ['mask', 'contour']
        self.vis_mode = vis_mode
        self.contour_size = contour_size

    def show_mask(self, img:np.ndarray, masks:np.ndarray, labels:np.ndarray):
        # cat_mask = cv2.imread('data/lmo/test/000002/mask/000243_000004.png', cv2.IMREAD_GRAYSCALE)
        height, width = img.shape[0], img.shape[1]
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for mask, label in zip(masks, labels):
            color = self.colors[label]
            colored_mask[mask] = color
        # points_y, points_x = np.nonzero(cat_mask)
        # top, bottom = points_y.min(), points_y.max()
        # left, right = points_x.min(), points_x.max()
        # colored_mask[top-5:bottom+5, left-5:right+5] = (0, 0, 0)
        # colored_mask[cat_mask>0] = (0, 0, 0)
        new_img = img * 0.5 + colored_mask*0.5
        return new_img

    def show_contour(self, img:np.ndarray, masks:np.ndarray, labels:np.ndarray):
        for mask, label in zip(masks, labels):
            contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            img = cv2.drawContours(img, contours, -1, self.colors[label], self.contour_size)
        return img


    def __call__(self, img:np.ndarray, rotations:np.ndarray, translations:np.ndarray, labels:np.ndarray, camera_k:np.ndarray, scores:Optional[np.ndarray]=None, out_file=None):
        if rotations.size == 0:
            return img
        rotations, translations = torch.from_numpy(rotations).to('cuda'), torch.from_numpy(translations).to('cuda')
        labels, camera_k = torch.from_numpy(labels).to('cuda'), torch.from_numpy(camera_k).to('cuda')
        if scores is not None:
            inds = torch.from_numpy(scores).to('cuda') > self.score_thr
            rotations, translations = rotations[inds], translations[inds]
            labels, camera_k = labels[inds], camera_k[inds]
        
        img = np.ascontiguousarray(img)

        render_outputs = self.render.forward(rotations, translations, camera_k, labels)
        rendered_img, fragments = render_outputs['images'], render_outputs['fragments']
        rendered_depths = fragments.zbuf
        rendered_depths = rendered_depths[..., 0]
        rendered_masks = (rendered_depths > 0).cpu().numpy()
        if self.vis_mode == 'mask':
            output = self.show_mask(img, rendered_masks, labels)
        else:
            output = self.show_contour(img, rendered_masks, labels)
        if out_file is not None:
            mmcv.imwrite(output, out_file)
        return output
    