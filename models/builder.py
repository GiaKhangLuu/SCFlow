# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from registry import MODELS

BACKBONES = MODELS
LOSSES = MODELS

def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)

def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)