import torch
import numpy as np
import os.path as osp
from typing import Optional, Dict, Sequence

from registry import HOOKS
from mmengine.dist import master_only
from mmengine.hooks import LoggerHook
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils import digit_version

class TensorboardLoggerHook(LoggerHook):
    """Class to log metrics to Tensorboard.

    Args:
        log_dir (string): Save directory location. Default: None. If default
            values are used, directory location is ``runner.work_dir``/tf_logs.
        interval (int): Logging interval (every k iterations). Default: True.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
    """

    def __init__(self,
                 log_dir: Optional[str] = None,
                 interval: int = 10,
                 ignore_last: bool = True,
                 by_epoch: bool = True):
        super().__init__(
            interval=interval, 
            ignore_last=ignore_last, 
            log_metric_by_epoch=by_epoch
        )
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        if (TORCH_VERSION == 'parrots'
                or digit_version(TORCH_VERSION) < digit_version('1.1')):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

    @master_only
    def after_run(self, runner) -> None:
        self.writer.close()

@HOOKS.register_module()
class TensorboardImgLoggerHook(TensorboardLoggerHook):
    def __init__(self,
                log_dir=None,
                interval=10,
                ignore_last=True,
                by_epoch=True,
                image_format='CHW'):
        super().__init__(
            log_dir=log_dir, 
            interval=interval, 
            ignore_last=ignore_last, 
            by_epoch=by_epoch)
        self.image_format = image_format
    
    @master_only
    def log(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))
        
        if 'log_imgs' in runner.outputs:
            log_imgs = runner.outputs['log_imgs']
            for tag, val in log_imgs.items():
                if isinstance(val, torch.Tensor):
                    self.writer.add_image(tag, val, self.get_iter(runner), dataformats=self.image_format)
                elif isinstance(val, np.ndarray):
                    if val.dtype == np.uint8:
                        val = val.astype(np.float32) / 255
                    elif val.dtype == np.float32:
                        pass 
                    else:
                        raise RuntimeError(f"Expect np.ndarray to be in np.flot32 or np.uint8 dtype, but got {val.dtype}")
                    self.writer.add_image(tag, val, self.get_iter(runner), dataformats=self.image_format)
                elif isinstance(val, (list, tuple)):
                    if isinstance(val[0], torch.Tensor):
                        val = torch.stack(val, dim=0)
                    elif isinstance(val[0], np.ndarray):
                        val = np.stack(val, axis=0)
                    else:
                        raise RuntimeError(f'Unexpected data type:{type(val[0])}')
                    self.writer.add_images(tag, val, self.get_iter(runner), dataformats='N'+self.image_format)
                else:
                    raise RuntimeError(f'Unexpected data type:{type(val)}')
