import csv
import json
from collections import OrderedDict
import time


def list_chunk(lst, n):
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def get_area(minx, miny, maxx, maxy):
    area = round((float(maxy) - float(miny)) * (float(maxx) - float(minx)), 2)
    return area


"""

config0

"""

csv_path = "/opt/ml/example.csv"
final_complete_path = "/opt/ml/dataset/complete2.json"

inference_json_path = "/opt/ml/example_to_json.json"
score_threshold = 0.9
numof_train_annotation = 23144
numof_train_images = 4883


data = list()
f = open(csv_path, "r")
rea = csv.reader(f)
cnt = 0
annotations = []
annotid = 0
for row in rea:
    stlist = list(row[0].split())
    list_chunked = list_chunk(stlist, 6)
    cnt += 1
    if cnt == 1:
        continue
    for i in range(len(list_chunked)):
        category = list_chunked[i][0]
        score = float(list_chunked[i][1])
        bbox_info = [
            round(float(list_chunked[i][2]), 1),
            round(float(list_chunked[i][3]), 1),
            round(float(list_chunked[i][4]), 1),
            round(float(list_chunked[i][5]), 1),
        ]
        image_id = int(str(row[1])[-8:-4]) + numof_train_images
        area = get_area(bbox_info[0], bbox_info[1], bbox_info[2], bbox_info[3])

        if score > score_threshold:
            annotation = {
                "image_id": image_id,
                "category_id": int(category),
                "area": float(area),
                "bbox": bbox_info,
                "iscrowd": 0,
                "id": annotid + numof_train_annotation,
            }
            annotations.append(annotation)
            annotid += 1

json_data = {"annotations": annotations}

with open(inference_json_path, "w") as json_file:
    json.dump(json_data, json_file, indent=4)

f.close

time.sleep(5)


"""
    
    images의 id를 원래의 id에 numof_train_images를 더해서 새롭게 저장하는 코드
    
"""


"""

config1

"""
# JSON 파일 경로와 수정할 key, 새로운 값 지정
test_file_path = "/opt/ml/dataset/test.json"
manipulate_test_path = "/opt/ml/dataset/manipaulate_test.json"


def change_filename(file_path, output_path):
    # JSON 파일 불러오기
    with open(file_path, "r") as file:
        data = json.load(file)

    for key, value in data.items():
        if key == "images":
            for val in value:
                val["id"] = val["id"] + numof_train_images

    # /opt/ml/dataset/trainparent/train
    # 수정된 JSON 파일 저장
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)


change_filename(test_file_path, manipulate_test_path)

time.sleep(5)

"""

example_to_json.json과 manipulate_test.json을 merge하는 코드

"""
"""

config2

"""

testmergeinfer_file_path = "/opt/ml/new_test.json"


def merge_json(first_path, second_path):
    merged_json = {}
    with open(first_path, "r") as file:
        data1 = json.load(file)
    with open(second_path, "r") as file:
        data2 = json.load(file)
    # json1의 키와 값을 병합
    for key, value in data1.items():
        if key != "annotations":
            merged_json[key] = value

    # json2의 키와 값을 추가
    for key, value in data2.items():
        if key == "annotations":
            merged_json[key] = value

    return merged_json


# JSON 병합
merged_json = merge_json(manipulate_test_path, inference_json_path)

# 결과를 새로운 JSON 파일로 저장
with open(testmergeinfer_file_path, "w") as file:
    json.dump(merged_json, file, indent=4)

time.sleep(5)

"""위에까지 수정 완료 아래부터는 Mergeinferoverlap"""

"""
train.json과 new_test.json(testmergeinfer_file_path)를 merge 하는 코드

"""

"""
config3
"""


origin_train_path = "/opt/ml/dataset/train.json"


def mergefinal_json(json1, json2):
    merged_json = {}
    for key, value in json1.items():
        if key == "categories":
            merged_json[key] = value
            continue
        if isinstance(value, list) and isinstance(json2[key], list):
            merged_json[key] = value + json2[key]
        else:
            merged_json[key] = value
        print("key", len(merged_json[key]))

    return merged_json


# 첫 번째 JSON 파일 로드
with open(origin_train_path, "r") as file:
    json1 = json.load(file)

# 두 번째 JSON 파일 로드
with open(testmergeinfer_file_path, "r") as file:
    json2 = json.load(file)

# JSON 병합
final_json = mergefinal_json(json1, json2)

# 결과를 새로운 JSON 파일로 저장
with open(final_complete_path, "w") as file:
    json.dump(final_json, file, indent=4)


"""
    사용법
    1. csv_path 경로 설정하기
    2. final_complete_path 경로 설정해주기 (따로 안바꿔줘도 됨)
    3. 아래 명령어 돌리기
    python /opt/ml/level2_objectdetection-cv-19/codebook/csvtodata.py
    4. 아래 명령어 다 실행되어 생긴 final_complete_path 를 해당 config에 train_annotation 부분에 넣어주기
    ex. final_complete_path = "/opt/ml/dataset/complete2.json" 이면
    학습할때 configs/custom/MM_baseline_train40.py 쓴다고 가정하면
    train_annotation = "complete2.json" 이렇게 바꿔주고 학습 하기
    5. 로깅 찍히는 게 train 시 [50/1866] 인지 확인하기
"""
