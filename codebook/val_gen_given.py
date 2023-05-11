# code for generation validation set

import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
from collections import Counter

# load json: modify the path to your own ‘train.json’ file
annotation = '../dataset/train.json'

with open(annotation) as f: 
    data = json.load(f)

var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
X = np.ones((len(data['annotations']),1))
y = np.array([v[1] for v in var])
groups = np.array([v[0] for v in var])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)

# for train_idx, val_idx in cv.split(X, y, groups):
#     print("TRAIN:", groups[train_idx])
#     print(" ", y[train_idx])
#     print(" TEST:", groups[val_idx])
#     print(" ", y[val_idx])
    

def get_distribution(y):
    y_distr = Counter(y)
    y_vals_sum = sum(y_distr.values())

    return [f'{y_distr[i]/y_vals_sum:.2%}' for i in range(np.max(y) +1)]

distrs = [get_distribution(y)]
index = ['training set']

for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X,y, groups)):
    train_y, val_y = y[train_idx], y[val_idx]
    train_gr, val_gr = groups[train_idx], groups[val_idx]

    assert len(set(train_gr) & set(val_gr)) == 0 
    distrs.append(get_distribution(train_y))

    distrs.append(get_distribution(val_y))
    index.append(f'train - fold{fold_ind}')
    index.append(f'val - fold{fold_ind}')

categories = [d['name'] for d in data['categories']]
df = pd.DataFrame(distrs, index=index, columns = [categories[i] for i in range(np.max(y) + 1)])
print(df)