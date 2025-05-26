from mmengine.registry import Registry, build_from_cfg

MODELS = Registry('MODELS')

def build_decoder(cfg):
    return build_from_cfg(cfg, MODELS)