import os
import json
import pandas as pd
import subprocess

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


tr_file_name = ['clean_30_train_fold1.json', 'clean_35_train_fold1.json', 'clean_40_train_fold1.json', 'val_fold1.json']

img_path = '/opt/ml/dataset/'
config_path = '/opt/ml/dataset_yolo/'

path_list = []
for fname in tr_file_name:
    coco_path = os.path.join(config_path, fname)
    coco_train = COCO(coco_path)
    df = df_gen(coco_train)

    img_series = df.groupby('image_id')["class_id"].apply(lambda x: len(x))
    image_id = list(i.split('/')[-1] for i in img_series.index)
    #print(len(image_id))

    ## img 저장
    if 'train' in fname:
        ori_root = os.path.join(img_path, 'train')
        if '30' in fname:
            save_root = os.path.join(config_path, 'images/train30')
            os.makedirs(save_root, exist_ok=True)
        elif '35' in fname:    
            save_root = os.path.join(config_path, 'images/train35')
            os.makedirs(save_root, exist_ok=True)
        elif '40' in fname:    
            save_root = os.path.join(config_path, 'images/train40') 
            os.makedirs(save_root, exist_ok=True)
    else:
        ori_root = os.path.join(img_path, 'train')
        save_root = os.path.join(config_path, 'images/valid')
        os.makedirs(save_root, exist_ok=True)
    
    path_list.append(save_root)
    for i in image_id:
        ori_img_root = os.path.join(ori_root, i)
        save_img_root = os.path.join(save_root, i)
        print(f'cp {ori_img_root} {save_img_root}')
        subprocess.call(f'cp {ori_img_root} {save_img_root}', shell=True)

for i in path_list:
    print(i, len(os.listdir(i)))
