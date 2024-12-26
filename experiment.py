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
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    return img

def main():
    resized = False
    desnow_all_path = 'drive_dataset/desnow_all' if not resized else 'drive_dataset/desnow_all_resized'
    desnow_part_path = 'drive_dataset/desnow_part' if not resized else 'drive_dataset/desnow_part_resized'
    desnow_all_saved_path = 'drive_dataset/desnow_all_pred' if not resized else 'drive_dataset/desnow_all_resized_pred'
    desnow_part_saved_path = 'drive_dataset/desnow_part_pred' if not resized else 'drive_dataset/desnow_part_resized_pred'
    gt_bbox_path = 'drive_dataset/bbox_processed' if not resized else 'drive_dataset/bbox_resized_processed'
    gt_data_path = 'drive_dataset/clear' if not resized else 'drive_dataset/clear_resized'
    gt_data_saved_path = 'drive_dataset/clear_gt' if not resized else 'drive_dataset/clear_resized_gt'
    
    bbox_postfix = 'gtBbox3d.json'
    img_postfix = 'leftImg8bit.jpg'
    gt_img_postfix = 'leftImg8bit.png'
    desnow_all_iou = 0
    desnow_part_iou = 0
    cnt = 0
    os.makedirs(gt_data_saved_path, exist_ok=True)
    os.makedirs(desnow_all_saved_path, exist_ok=True)
    os.makedirs(desnow_part_saved_path, exist_ok=True)
    for data in tqdm(os.listdir(gt_bbox_path)):
        if not data.endswith(bbox_postfix):
            continue
        with open(os.path.join(gt_bbox_path, data), 'r') as f:
            gt_bbox_list = json.load(f)
        if 'car' not in gt_bbox_list.keys():
            continue
        desnow_all_img_path = os.path.join(desnow_all_path, data.replace(bbox_postfix, img_postfix))
        desnow_part_img_path = os.path.join(desnow_part_path, data.replace(bbox_postfix, img_postfix))
        desnow_all_bbox_list = detect_objects(desnow_all_img_path)
        desnow_part_bbox_list = detect_objects(desnow_part_img_path)
        gt_img_path = os.path.join(gt_data_path, data.replace(bbox_postfix, gt_img_postfix))
        gt_img = draw_bbox(cv2.imread(gt_img_path), gt_bbox_list['car'])
        cv2.imwrite(os.path.join(gt_data_saved_path, data.replace(bbox_postfix, img_postfix)), gt_img)
        
        desnow_all_img = draw_bbox(cv2.imread(desnow_all_img_path), desnow_all_bbox_list.get('car', []))
        desnow_part_img = draw_bbox(cv2.imread(desnow_part_img_path), desnow_part_bbox_list.get('car', []))
        cv2.imwrite(os.path.join(desnow_all_saved_path, data.replace(bbox_postfix, img_postfix)), desnow_all_img)
        cv2.imwrite(os.path.join(desnow_part_saved_path, data.replace(bbox_postfix, img_postfix)), desnow_part_img)
        for bbox in gt_bbox_list['car']:
            desnow_all_intersection, desnow_all_union = 0, 0
            desnow_part_intersection, desnow_part_union = 0, 0
            for desnow_all_bbox in desnow_all_bbox_list.get('car', []):
                intersection, union = calculate_iou(bbox, desnow_all_bbox)
                desnow_all_intersection += intersection
                desnow_all_union += union
            desnow_all_iou += desnow_all_intersection / desnow_all_union if desnow_all_union != 0 else 0
            for desnow_part_bbox in desnow_part_bbox_list.get('car', []):
                intersection, union = calculate_iou(bbox, desnow_part_bbox)
                desnow_part_intersection += intersection
                desnow_part_union += union
            desnow_part_iou += desnow_part_intersection / desnow_part_union if desnow_part_union != 0 else 0
            cnt += 1
    print('Desnow all IoU:', desnow_all_iou / cnt)
    print('Desnow part IoU:', desnow_part_iou / cnt)
                    
if __name__ == "__main__":
    main()