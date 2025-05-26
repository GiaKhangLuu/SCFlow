from mmengine import Config
from registry import MODELS, DATASETS, HOOKS, METRICS
from mmengine.runner import Runner
from mmengine.optim import DefaultOptimWrapperConstructor
from torch.nn.parallel import DataParallel as MMDataParallel

cfg = Config.fromfile("./configs/refine_models/scflow_lumi_piano_real.py")

model = MODELS.build(cfg.model)
metrics = METRICS.build(cfg.val_evaluator)

dataset = DATASETS.build(cfg.test_dataloader.dataset)

result_dict = dataset[0]
print(result_dict)

#print(model.cuda)
#model = model.to("cuda")

#print(cfg.get('load_from'))

#runner = Runner.from_cfg(cfg)

#runner.train()

#optim_wrapper_builder = DefaultOptimWrapperConstructor(cfg.optim_wrapper)
#optim_wrapper = optim_wrapper_builder(model)

#print(optim_wrapper)

#print(dataset.getitem(0)["img"].data.shape)
