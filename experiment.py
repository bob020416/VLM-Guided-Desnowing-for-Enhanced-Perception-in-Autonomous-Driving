import argparse
import torch
import cv2
import os
import json
from tqdm import tqdm
# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(img_path):
    img = cv2.imread(img_path)
    results = model(img)
    predictions = results.xyxy[0].cpu().numpy()
    bbox_list = {}
    for *box, conf, cls_id in predictions:
        if int(cls_id) == 2:
            label = 'person' if int(cls_id) == 0 else 'car'
            if label not in bbox_list.keys():
                bbox_list[label] = []
            box = list(map(int, box))
            bbox_list[label].append(box)
            
    return bbox_list

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    x_overlap = max(0, min(x2, x4) - max(x1, x3))
    y_overlap = max(0, min(y2, y4) - max(y1, y3))
    intersection = x_overlap * y_overlap
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
    return intersection, union

def draw_bbox(img, bbox_list):
    for bbox in bbox_list:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
    return img
        
def parse_args():
    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('-r', '--resized', action='store_true', help='Whether to use resized images')
    parser.add_argument('-m', '--metric', type=int, default=1, help='Which metric to use')
    return parser.parse_args()

def main():
    args = parse_args()
    resized = args.resized
    metric = args.metric
    desnow_whole_path = 'drive_dataset/desnow_all' if not resized else 'drive_dataset/desnow_all_resized'
    desnow_part_path = 'drive_dataset/desnow_part' if not resized else 'drive_dataset/desnow_part_resized'
    desnow_whole_saved_path = 'drive_dataset/desnow_all_pred' if not resized else 'drive_dataset/desnow_all_resized_pred'
    desnow_part_saved_path = 'drive_dataset/desnow_part_pred' if not resized else 'drive_dataset/desnow_part_resized_pred'
    gt_bbox_path = 'drive_dataset/bbox_processed' if not resized else 'drive_dataset/bbox_resized_processed'
    clear_data_path = 'drive_dataset/clear' if not resized else 'drive_dataset/clear_resized'
    gt_data_saved_path = 'drive_dataset/clear_gt' if not resized else 'drive_dataset/clear_resized_gt'
    
    bbox_postfix = 'gtBbox3d.json'
    img_postfix = 'leftImg8bit.jpg'
    gt_img_postfix = 'leftImg8bit.png'
    cnt = 0
    if metric == 1 or metric == 2:
        desnow_whole_iou = 0
        desnow_part_iou = 0
        clear_iou = 0
    elif metric == 3:
        total_gt = 0
        desnow_whole_pred = 0
        desnow_part_pred = 0
        clear_pred = 0
        accurate_treshold = 0.5
    os.makedirs(gt_data_saved_path, exist_ok=True)
    os.makedirs(desnow_whole_saved_path, exist_ok=True)
    os.makedirs(desnow_part_saved_path, exist_ok=True)
    for data in tqdm(os.listdir(gt_bbox_path)):
        if not data.endswith(bbox_postfix):
            continue
        with open(os.path.join(gt_bbox_path, data), 'r') as f:
            gt_bbox_list = json.load(f)
        if 'car' not in gt_bbox_list.keys():
            continue
        desnow_whole_img_path = os.path.join(desnow_whole_path, data.replace(bbox_postfix, img_postfix))
        desnow_part_img_path = os.path.join(desnow_part_path, data.replace(bbox_postfix, img_postfix))
        desnow_whole_bbox_list = detect_objects(desnow_whole_img_path)
        desnow_part_bbox_list = detect_objects(desnow_part_img_path)
        
        clear_img_path = os.path.join(clear_data_path, data.replace(bbox_postfix, gt_img_postfix))
        clear_bbox_list = detect_objects(clear_img_path)
        
        gt_img = draw_bbox(cv2.imread(clear_img_path), gt_bbox_list['car'])
        cv2.imwrite(os.path.join(gt_data_saved_path, data.replace(bbox_postfix, img_postfix)), gt_img)
        
        desnow_whole_img = draw_bbox(cv2.imread(desnow_whole_img_path), desnow_whole_bbox_list.get('car', []))
        desnow_part_img = draw_bbox(cv2.imread(desnow_part_img_path), desnow_part_bbox_list.get('car', []))
        cv2.imwrite(os.path.join(desnow_whole_saved_path, data.replace(bbox_postfix, img_postfix)), desnow_whole_img)
        cv2.imwrite(os.path.join(desnow_part_saved_path, data.replace(bbox_postfix, img_postfix)), desnow_part_img)
        
        for bbox in gt_bbox_list['car']:
            if metric == 1:
                desnow_whole_intersection, desnow_whole_union = 0, 0
                desnow_part_intersection, desnow_part_union = 0, 0
                clear_intersection, clear_union = 0, 0
            elif metric == 3:
                total_gt += 1
                
            for desnow_whole_bbox in desnow_whole_bbox_list.get('car', []):
                intersection, union = calculate_iou(bbox, desnow_whole_bbox)
                if metric == 1:
                    desnow_whole_intersection += intersection
                    desnow_whole_union += union
                elif metric == 2:
                    desnow_whole_iou += intersection / union if union != 0 else 0
                elif metric == 3:
                    iou = intersection / union if union != 0 else 0
                    if iou >= accurate_treshold:
                        desnow_whole_pred += 1
                        break
            for desnow_part_bbox in desnow_part_bbox_list.get('car', []):
                intersection, union = calculate_iou(bbox, desnow_part_bbox)
                if metric == 1:
                    desnow_part_intersection += intersection
                    desnow_part_union += union
                elif metric == 2:
                    desnow_part_iou += intersection / union if union != 0 else 0
                elif metric == 3:
                    iou = intersection / union if union != 0 else 0
                    if iou >= accurate_treshold:
                        desnow_part_pred += 1
                        break
            
            for clear_bbox in clear_bbox_list.get('car', []):
                intersection, union = calculate_iou(bbox, clear_bbox)
                if metric == 1:
                    clear_intersection += intersection
                    clear_union += union
                elif metric == 2:
                    clear_iou += intersection / union if union != 0 else 0
                elif metric == 3:
                    iou = intersection / union if union != 0 else 0
                    if iou >= accurate_treshold:
                        clear_pred += 1
                        break
            if metric == 1:
                desnow_whole_iou += desnow_whole_intersection / desnow_whole_union if desnow_whole_union != 0 else 0
                desnow_part_iou += desnow_part_intersection / desnow_part_union if desnow_part_union != 0 else 0
                clear_iou += clear_intersection / clear_union if clear_union != 0 else 0
            cnt += 1
    if metric == 1 or metric == 2:
        print('Clear IoU:', clear_iou / cnt)
        print('Desnow whole IoU:', desnow_whole_iou / cnt)
        print('Desnow part IoU:', desnow_part_iou / cnt)
    elif metric == 3:
        print('Clear precision:', clear_pred / total_gt)
        print('Desnow whole precision:', desnow_whole_pred / total_gt)
        print('Desnow part precision:', desnow_part_pred / total_gt)
if __name__ == "__main__":
    main()