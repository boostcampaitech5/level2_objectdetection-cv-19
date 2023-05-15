# get valication data
import csv
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
# check distribution
import pandas as pd
from collections import Counter
from torch.utils.data import Subset


def get_anotations_by_image(annotations, image_idx):
    anns = []
    for ann in annotations:
        if ann['image_id'] in image_idx:
            anns.append(ann)
    return anns


def compute_overlap(boxes, query_boxes):
    """
    Args:
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
    Returns:
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0]) *
            (query_boxes[k, 3] - query_boxes[k, 1])
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0])
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1])
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0]) *
                        (boxes[n, 3] - boxes[n, 1]) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def get_real_annotations(table):
    res = dict()
    ids = table['ImageID'].values.astype(str)
    labels = table['LabelName'].values.astype(str)
    xmin = table['XMin'].values.astype(np.float32)
    xmax = table['XMax'].values.astype(np.float32)
    ymin = table['YMin'].values.astype(np.float32)
    ymax = table['YMax'].values.astype(np.float32)

    for i in range(len(ids)):
        id = ids[i]
        label = labels[i]
        if id not in res:
            res[id] = dict()
        if label not in res[id]:
            res[id][label] = []
        box = [xmin[i], ymin[i], xmax[i], ymax[i]]
        res[id][label].append(box)

    return res


def get_detections(table):
    res = dict()
    ids = table['ImageID'].values.astype(str)
    labels = table['LabelName'].values.astype(str)
    scores = table['Conf'].values.astype(np.float32)
    xmin = table['XMin'].values.astype(np.float32)
    xmax = table['XMax'].values.astype(np.float32)
    ymin = table['YMin'].values.astype(np.float32)
    ymax = table['YMax'].values.astype(np.float32)

    for i in range(len(ids)):
        id = ids[i]
        label = labels[i]
        if id not in res:
            res[id] = dict()
        if label not in res[id]:
            res[id][label] = []
        box = [xmin[i], ymin[i], xmax[i], ymax[i], scores[i]]
        res[id][label].append(box)

    return res


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


'''
    mean_average_precision_for_boxes: GT와 prediction을 넣으면 iou_threshold에 따라서 mAP를 계산해주는 함수.
    해당 대회에서는 mAP50을 사용하여 iou_threshold의 default값을 0.5로 설정
'''
def mean_average_precision_for_boxes(ann, pred, iou_threshold):

    #print(ann)
    #print(pred)

    LABEL_NAME = ["General trash", "Paper", "Paper pack", "Metal", 
              "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

    """
    :param ann: path to CSV-file with annotations or numpy array of shape (N, 6)
    :param pred: path to CSV-file with predictions (detections) or numpy array of shape (N, 7)
    :param iou_threshold: IoU between boxes which count as 'match'. Default: 0.5
    :return: tuple, where first value is mAP and second values is dict with AP for each class.
    """

    if isinstance(ann, str):
        valid = pd.read_csv(ann)
    else:
        valid = pd.DataFrame(ann, columns=['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax'])

    if isinstance(pred, str):
        preds = pd.read_csv(pred)
    else:
        preds = pd.DataFrame(pred, columns=['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax'])

    ann_unique = valid['ImageID'].unique()
    preds_unique = preds['ImageID'].unique()

    #print('Number of files in annotations: {}'.format(len(ann_unique)))
    #print('Number of files in predictions: {}'.format(len(preds_unique)))

    unique_classes = valid['LabelName'].unique().astype(str)
    #print('Unique classes: {}'.format(len(unique_classes)))

    #print(".")
    all_detections = get_detections(preds)
    all_annotations = get_real_annotations(valid)
    #print('all Detections : ',all_detections)
    #print('all Annotations : ',all_annotations)

    average_precisions = {}
    for _, label in enumerate(sorted(unique_classes)):

        # Negative class
        if str(label) == 'nan':
            continue

        false_positives = []
        true_positives = []
        scores = []
        num_annotations = 0.0

        for i in range(len(ann_unique)):
            #print("ann_unique",ann_unique)
            #print("all_detection",all_detections)
            detections = []
            annotations = []
            id = ann_unique[i]
            if id in all_detections:
                if label in all_detections[id]:
                    detections = all_detections[id][label]
            if id in all_annotations:
                if label in all_annotations[id]:
                    annotations = all_annotations[id][label]

            #print('detections :', detections)
            #print('annotations :', annotations)

            if len(detections) == 0 and len(annotations) == 0:
                continue

            num_annotations += len(annotations)
            detected_annotations = []

            annotations = np.array(annotations, dtype=np.float64)
            for d in detections:
                scores.append(d[4])

                if len(annotations) == 0:
                    false_positives.append(1)
                    true_positives.append(0)
                    continue

                overlaps = compute_overlap(np.expand_dims(np.array(d, dtype=np.float64), axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives.append(0)
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives.append(1)
                    true_positives.append(0)

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        false_positives = np.array(false_positives)
        true_positives = np.array(true_positives)
        scores = np.array(scores)

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        # 각 label 별로 PR curve plot
        #plt.plot(recall, precision)
        #plt.title(f'{LABEL_NAME[int(label)]} PR curve', fontsize=20)
        #plt.xlabel('Recall', fontsize=18)
        #plt.ylabel('Precision', fontsize=18)
        #plt.show()

        # compute average precision
        #print(recall, precision)
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
        s1 = "{:10s} | {:.6f} | {:7d}".format(LABEL_NAME[int(label)], average_precision, int(num_annotations))
        #print(s1)
    
    #print(average_precisions)


    present_classes = 0
    precision = 0
    #print(average_precisions.items())
    for label, (average_precision, num_annotations) in average_precisions.items():
        if num_annotations > 0:
            present_classes += 1
            precision += average_precision
            
    mean_ap = precision / present_classes
    #print('mAP: {:.6f}'.format(mean_ap))
    
    return mean_ap, average_precisions


def get_incorrect_data(dir_true_json, dir_pred_csv, dir_save_json, iou_threshold, criterion):

    # load json: modify the path to your own ‘dir_true_json’ file
    with open(dir_true_json) as f:
        # print(data)
        data = json.load(f)
        info = data['info']
        licenses = data['licenses']
        images = data['images']
        categories = data['categories']
        annotations = data['annotations']

    solution = {}
    for i in images:
        solution[i["id"]] = []

    n=0
    for ann in annotations:
        #print(ann["image_id"])
        if not solution[ann["image_id"]]:
            n=0

        solution[ann["image_id"]].append( [
            '1',
            ann['category_id'],
            ann["bbox"][0], 
            ann["bbox"][0]+ann["bbox"][2],
            ann["bbox"][1],
            ann["bbox"][1]+ann["bbox"][3] 
            ] )
        
        n+=1

    #print(solution)

    # solution  = [  
    #               0: [[idx, Class,1, a,b,c,d], [idx, Class,1, a,b,c,d], [idx, Class,1, a,b,c,d]],
    #               1: [[idx, Class,1, a,b,c,d], [idx, Class,1, a,b,c,d]                        ]] 
    #              ]

    answer = {}
    f = open(dir_pred_csv, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    a =1
    for line in rdr:
        if a:
            a=0
            continue

        key = int(line[1][-8:-4])
        answer[key]=[]
        ln = list(map(float, line[0].split()))
        n=0
        for t in range(len(ln)//6):
            answer[key].append( ['1'] + [int(ln[6*t])] +[ln[6*t+1]] +[ln[6*t+2]] +[ln[6*t+4]] +[ln[6*t+3]] +[ln[6*t+5]])
            n+=1
    f.close()

    # answer  = [  
    #               0: [[idx, Class,p, a,b,c,d], [idx, Class,p, a,b,c,d], [idx, Class,p, a,b,c,d]],
    #               1: [[idx, Class,p, a,b,c,d], [idx, Class,p, a,b,c,d]                        ]] 
    #            ]

    #print(answer)

    incorrect_list = []
    for key in answer.keys():
        pred_boxes = answer[key]
        true_boxes = solution[key]
        n_class = len(categories)
        mean_ap, average_precisions = mean_average_precision_for_boxes(true_boxes, pred_boxes, iou_threshold)

        #print(mean_ap)
        if mean_ap < criterion:
            incorrect_list.append(key)

    #print(incorrect_list)
    #print(len(incorrect_list))


    new_images = []
    for idx in range(len(incorrect_list)):
        new_images.append(images[idx])

    with open(dir_save_json,'w') as train_writer:
            json.dump({
                'info' : info,
                'licenses' : licenses,
                'images' : new_images,
                'categories' : categories,
                'annotations' : get_anotations_by_image(annotations, incorrect_list)

            }, train_writer, indent=2)



    print('\nCreating %s... Done !' %dir_save_json)
    print(".")



if __name__ == '__main__':
    dir_true_json = './kfold3_val.json'
    dir_pred_csv = './submission_fold3.csv'
    dir_save_json = './incorrect.json'
    iou_threshold = 0.5
    criterion = 0.4
    get_incorrect_data(dir_true_json, dir_pred_csv, dir_save_json, iou_threshold, criterion)