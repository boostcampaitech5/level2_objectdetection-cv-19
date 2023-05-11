import json
import os

from pycocotools.coco import COCO
from test_code import df_gen

coco_path = './dataset/35_train_fold1.json'
coco_train = COCO(coco_path)   #train_fold1

df = df_gen(coco_train)
num_series = df.groupby('image_id')["class_id"].apply(lambda x: len(x)).sort_values(ascending=False)
#print(num_series)

std = 30
save_name = './dataset/' + str(std) + '_train_fold1.json'
image_id_to_remove = list(int(i.split('/')[-1][:4]) for i in num_series[num_series>=std].index)  # 삭제하고자 하는 이미지 ID
print(image_id_to_remove)

with open(coco_path, 'r') as f:
    coco_data = json.load(f)

# 이미지 삭제
for image in coco_data['images']:
    if image['id'] in image_id_to_remove:
        #print(image['id'])
        coco_data['images'].remove(image)
        break

# 어노테이션 삭제
for annotation in coco_data['annotations']:
    if annotation['image_id'] in image_id_to_remove:
        #print(annotation['image_id'])
        coco_data['annotations'].remove(annotation)

# JSON 파일로 저장
with open(save_name, 'w') as f:
    json.dump(coco_data, f)
    