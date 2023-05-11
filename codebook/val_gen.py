import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Load COCO annotations file
with open('./dataset/coco_annotations.json', 'r') as f:
    annotations = json.load(f)


# Extract bounding box areas and classes
areas = []
classes = []
for annotation in annotations['annotations']:
    areas.append(annotation['bbox'][2] * annotation['bbox'][3])
    classes.append(annotation['category_id'])

# Convert areas to numpy array for easier manipulation
areas = np.array(areas)

# Perform stratified k-fold splitting based on area and class
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, val_index) in enumerate(skf.split(areas, classes)):
    # Create train and validation sets
    train_ann = {'images': [], 'annotations': [], 'categories': annotations['categories']}
    val_ann = {'images': [], 'annotations': [], 'categories': annotations['categories']}

    # Add images and annotations to corresponding set
    tr_img_lst = set()
    val_img_lst = set()
    for index, ann in enumerate(annotations['annotations']):
        image_id = ann['image_id'] # annotation을 차례로 가져옴
        #print('train_idx', train_index)
        #print('val_idx', val_index)
        if image_id in train_index:
            if image_id not in tr_img_lst:
                tr_img_lst.add(image_id) # 겹치는 idx면 images에 저장하지 않도록
                train_ann['images'].append(annotations['images'][image_id])
            train_ann['annotations'].append(ann)
        elif image_id in val_index:
            if image_id not in val_img_lst:
                val_img_lst.add(image_id)
                val_ann['images'].append(annotations['images'][image_id])
            val_ann['annotations'].append(ann)

    # Save train and validation sets as separate json files
    with open(f'train_fold{fold}.json', 'w') as f:
        json.dump(train_ann, f)
    with open(f'val_fold{fold}.json', 'w') as f:
        json.dump(val_ann, f)
