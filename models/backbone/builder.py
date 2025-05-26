from mmengine.registry import build_from_cfg, Registry

MODELS = Registry('MODELS')

def build_backbone(cfg):
    return build_from_cfg(cfg, MODELS)