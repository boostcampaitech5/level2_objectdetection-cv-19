import os
import json
import pandas as pd

from pycocotools.coco import COCO


def df_gen(coco_obj):
    df = pd.DataFrame()
    image_ids = []
    class_name = []
    class_id = []
    x_min = []
    y_min = []
    x_max = []
    y_max = []
    classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

    for image_id in coco_obj.getImgIds():                       # category ID를 input으로 그에 해당하는 Image ID를 return
        imageinfo = coco_obj.loadImgs(image_id)[0]              # Image ID를 input으로 annotations의 image dict 전체(상세정보) return 
        ann_ids = coco_obj.getAnnIds(imgIds=imageinfo['id'])
        anns = coco_obj.loadAnns(ann_ids)
        filename = imageinfo['file_name']

        for ann in anns:
            image_ids.append(filename)
            class_name.append(classes[ann['category_id']])
            class_id.append(ann['category_id'])
            x_min.append(float(ann['bbox'][0]))
            y_min.append(float(ann['bbox'][1]))
            x_max.append(float(ann['bbox'][0]) + float(ann['bbox'][2]))
            y_max.append(float(ann['bbox'][1]) + float(ann['bbox'][3]))

    df['image_id'] = image_ids
    df['class_name'] = class_name
    df['class_id'] = class_id
    df['x_min'] = x_min
    df['y_min'] = y_min
    df['x_max'] = x_max
    df['y_max'] = y_max
    
    return df


coco_path = './dataset/clean_30_train_fold1.json' #'./dataset/tmp/train_fold1.json'
coco_train = COCO(coco_path)
df = df_gen(coco_train)

num_series = df.groupby('image_id')["class_id"].apply(lambda x: len(x)).sort_values(ascending=False)
#print(num_series)

std = 30
save_name = './dataset/clean_' + str(std) + '_train_fold1.json'
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
    