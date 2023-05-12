import requests
import os
import shutil
import subprocess
from dotenv import load_dotenv
import importlib
import sys
sys.dont_write_bytecode = True

# .env 로드 (HOST_NAME and Discord Hook URL)
load_dotenv("../.env")
discord_log_channel = os.environ.get("discord_log_channel")
discord_error_channel = os.environ.get("discord_error_channel")
HOST_NAME = os.environ.get("HOST_NAME")


# 결과 저장 폴더 생성 함수
def make_dir():
    if not os.path.exists("error_logs"):
        os.mkdir("error_logs")

    if not os.path.exists("completed"):
        os.mkdir("completed")


# 디스코드 알림 전송 함수
def discord_notice(channel, config, header, status="", error_msg=""):
    message = f"--------------------- {header} ---------------------\n"
    footer_size = len(message)
    message += f"EXP_NAME : {config.exp_name}\n"
    message += f"WORKER : {config.worker}\n"
    message += f"WORKING_HOST : {HOST_NAME}\n"
    if status:
        message += f"STATUS : {status}\n"
    if error_msg:
        message += f"ERROR_MESSAGE : \n {error_msg}"
    message += "-" * footer_size

    requests.post(channel, data={"content": message})


# 모델 학습 함수
def trainer(file_name):
    config = importlib.import_module(file_name[:-3])
    config = importlib.reload(config)
    print(config.exp_name)

    # 디스코드 알림 전송 (학습 시작)
    discord_notice(discord_log_channel, config, "TRAIN START")

    # 모델 학습 커맨드 실행
    command = f"python -B /opt/ml/level2_objectdetection-cv-19/mmdetection/tools/train.py {file_name}"
    process = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    process.wait()
    error_message = process.stderr.read().decode("EUC-KR", errors="ignore")
    
    # 결과 저장 및 디스코드 알림 전송 (에러 발생 여부에 따라)
    make_dir()
    if process.returncode == 0:
        shutil.move(file_name, f"completed/{config.exp_name}.py")
        discord_notice(discord_log_channel, config, "TRAIN FINISH", status="SUCCESS")
    else:
        if not os.path.exists(f"error_logs/{config.exp_name}"):
            os.mkdir(f"error_logs/{config.exp_name}")
        
        shutil.move(file_name, f"error_logs/{config.exp_name}/{config.exp_name}.py")
        
        with open(f"error_logs/{config.exp_name}/error_msg.txt", "w") as f:
            f.write(error_message)
            
        discord_notice(discord_log_channel, config, "TRAIN FINISH", status="FAILURE")
        discord_notice(discord_error_channel, config, "ERROR", error_msg=error_message)


# Message를 수신하고 수행할 작업 명시
def worker(message):
    # message를 yaml 파일로 저장
    with open("recieve.py", "w", encoding="utf-8") as f:
        f.write(message)
    
    # Model Trainer 호출
    trainer("recieve.py")