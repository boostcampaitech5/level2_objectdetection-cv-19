import os

BASE_PATH = "/opt/ml/level2_objectdetection-cv-19-develop/mmdetection"
EXP_BASE_PATH = 'opt/ml/output'
CONFIG_PATH = "configs/custom"

CONFIG_QUEUE_PATH = os.path.join(CONFIG_PATH, "queue")
CONFIG_ENDS_PATH = os.path.join(CONFIG_PATH, "ends")

configs = [file for file in os.listdir(CONFIG_QUEUE_PATH) if file != ".gitkeep"]

if configs:
    print(f"current config file: {configs[0]}") # base_config.py

    # train.py
    # os.system("git pull")
    os.system(f"python tools/train.py {os.path.join(CONFIG_QUEUE_PATH, configs[0])}")
    os.system(f"mv {os.path.join(CONFIG_QUEUE_PATH, configs[0])} {os.path.join(CONFIG_ENDS_PATH, configs[0])}")

    EXP_PATH = os.path.join(EXP_BASE_PATH, configs[0][:-3]) # opt/ml/output/base_config
    os.system(f"python inference.py --cfg_folder {os.path.join(EXP_PATH, configs[0])}")