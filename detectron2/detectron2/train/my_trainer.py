from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.train.my_mapper import MyMapper
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.train.loss_eval_hook import LossEvalHook
from detectron2.utils.util import ensure_dir

class MyTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(cfg, mapper = MyMapper, sampler = sampler) # cfg로부터 train dataloader return받음
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        ensure_dir(output_folder)
        
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
                     
    # def build_hooks(self):
    #     hooks = super().build_hooks()
    #     hooks.insert(-1,LossEvalHook(
    #         self.cfg.TEST.EVAL_PERIOD,
    #         self.model,
    #         build_detection_test_loader(
    #             self.cfg,
    #             self.cfg.DATASETS.TEST[0],
    #             DatasetMapper(self.cfg,True)
    #         )
    #     ))
    #     return hooks
    
    
    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #     if output_folder is None:
    #         os.makedirs('./output_eval', exist_ok = True)
    #         output_folder = './output_eval'
            
    #     return COCOEvaluator(dataset_name, cfg, False, output_folder)
