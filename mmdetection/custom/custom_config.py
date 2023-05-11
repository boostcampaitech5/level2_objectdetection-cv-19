from mmcv import Config
from mmdet.utils import get_device

def TrainCustomConfig(data_path='../../dataset/'):
    # base config file(faster_rcnn) 로드
    cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
    
    # 재활용 쓰레기 data의 class 지정
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    
    # dataset에 맞게 base config 수정
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = data_path
    cfg.data.train.ann_file = data_path + 'train.json' # train json 정보
    cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

    ''' 
    Q. 이 cfg.data.test 설정은 굳이 train에 없어도 되는 거였나?
    '''
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = data_path
    cfg.data.test.ann_file = data_path + 'test.json' # test json 정보
    cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

    cfg.data.samples_per_gpu = 4

    cfg.seed = 2022
    cfg.gpu_ids = [0]
    cfg.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'

    cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()

    return cfg

def TestCustomConfig(data_path='../../dataset/'):
    # base config file(faster_rcnn) 로드
    cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
    
    # 재활용 쓰레기 data의 class 지정
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # dataset에 맞게 base config 수정
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = data_path
    cfg.data.test.ann_file = data_path + 'test.json' # test json 정보
    cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
    cfg.data.test.test_mode = True

    cfg.data.samples_per_gpu = 4

    cfg.seed=2021
    cfg.gpu_ids = [1]
    cfg.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'

    cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    return cfg
