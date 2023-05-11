#!/usr/bin/env python
# coding: utf-8
import pandas as pd; pd.options.mode.chained_assignment = None
import numpy as np

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


### Dataframe generation
init_train = COCO('./dataset/coco_annotations.json')
coco_train = COCO('./dataset/35_train_fold1.json')   #train_fold0
coco_val   = COCO('./dataset/val_fold1.json')        #val_fold0  

init_df = df_gen(init_train)
train_df = df_gen(coco_train)
val_df = df_gen(coco_val)
print("\nInitialise dataset\n")


# ### Test Code
# 0. Class dist.
# 1. Total num of box : 23144
# 2. Total image num  : 4883 (0~4882.jpg)
# 3. Mean box area
    
def check_cls_dist(df_ori, df_train, df_val):
    dist1 = df_ori.class_name.value_counts().sort_index() / len(df_ori)*100
    dist2 = df_train.class_name.value_counts().sort_index() / len(df_train)*100
    dist3 = df_val.class_name.value_counts().sort_index() / len(df_val)*100
    distrs = [dist1, dist2, dist3]
    
    classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
    df = pd.DataFrame(distrs, index=['ori', 'tr', 'val'], columns=classes)
    print(f'{df}\n')


def check_num_box(df_ori, df_train, df_val):
    if len(df_ori.index) == len(df_train.index)+len(df_val.index):
        print(f'Total number of boxes : {len(df_ori.index)}')
        print(f'Train: {len(df_train.index)}, Val: {len(df_val.index)}')
        print('Correct!')
    else:
        print('Wrong Split result!')
        print(f'Total: {len(df_ori.index)}, Train: {len(df_train.index)}, Val: {len(df_val.index)}\n')


# 합쳐서 sort, 갯수랑 각 row 비교하면 됨 + 40개 이상인 갯수
def check_img_dup_and_cls(df_ori, df_train, df_val):
    se_ori = df_ori.groupby('image_id')["class_id"].unique().sort_index()
    se_train = df_train.groupby('image_id')["class_id"].unique()
    se_val = df_val.groupby('image_id')["class_id"].unique()
    se_concat = pd.concat([se_train, se_val]).sort_index()
    
    if len(se_concat) != len(se_ori):
        print('image duplication happened!\n')
    else:
        not_same_id = []
        for idx in se_ori.index:
            if len(se_ori[idx]) != len(se_concat[idx]):
                not_same_id.append(idx)
            elif not all(x in se_concat[idx] for x in se_ori[idx]):
                not_same_id.appned(idx)
        if not_same_id: 
            print(f'Different value found at {not_same_id}!\n')
        else:
            print('Same class number in each images! \nNo problem!\n')


def check_area(df_train, df_val):
    area_train = int(((df_train.x_max - df_train.x_min)*(df_train.y_max - df_train.y_min)).values.mean())
    area_val = int(((df_val.x_max - df_val.x_min)*(df_val.y_max - df_val.y_min)).values.mean())
    if abs(area_train-area_val)/area_train*100 >= 10.0:
        print("Area diff is too big!")
        print(f'train mean area: {area_train}, val mean area: {area_val}\n')
    else:
        print('No problem about area!\n')
    

def clean_dataset(df_train, df_val, std=30):
    tr_num  = (df_train.image_id.value_counts()>=std).sum()
    val_num = (df_val.image_id.value_counts()>=std).sum()
    print(f'The number of boxes over {std} (Total: {tr_num+val_num})')
    print(f'Train: {tr_num}, Val: {val_num}\n')


def check_dataset_avail(df_ori, df_train, df_val):
    check_cls_dist(df_ori, df_train, df_val)
    check_num_box(df_ori, df_train, df_val)
    check_img_dup_and_cls(df_ori, df_train, df_val)
    check_area(df_train, df_val)
    #clean_dataset(train_df, val_df, 40)


### Check whether something is wrong
check_dataset_avail(init_df, train_df, val_df)

clean_dataset(train_df, val_df, 40)
clean_dataset(train_df, val_df, 35)
clean_dataset(train_df, val_df, 30)

# train_fold1 기준으로 cleaning 진행하자!
# The number of boxes over 40 (Total: 14)
# Train: 10, Val: 4

# The number of boxes over 35 (Total: 25)
# Train: 19, Val: 6

# The number of boxes over 30 (Total: 57)
# Train: 46, Val: 11