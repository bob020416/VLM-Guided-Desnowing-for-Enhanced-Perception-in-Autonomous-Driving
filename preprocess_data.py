import os
import cv2
import random
import json
from tqdm import tqdm
correct_img_shape = cv2.imread('/home/hcis-s17/multimodal_manipulation/patrick/CV/final/CSD/Train/Snow/1.tif').shape

def preprocess_data():
    data_root = 'drive_dataset/clear'
    saved_data_root = 'drive_dataset/clear_resized'
    os.makedirs(saved_data_root, exist_ok=True)

    for img in tqdm(os.listdir(data_root)):
        img_path = os.path.join(data_root, img)
        img_data = cv2.imread(img_path)
        if img_data.shape != correct_img_shape:
            # crop correct shape from img_data in the center
            img_data = img_data[(img_data.shape[0] - correct_img_shape[0])//2:(img_data.shape[0] + correct_img_shape[0])//2, (img_data.shape[1] - correct_img_shape[1])//2:(img_data.shape[1] + correct_img_shape[1])//2]
            cv2.imwrite(os.path.join(saved_data_root, img), img_data)

def preprocess_bbox():
    data_root = 'drive_dataset/bbox'
    saved_data_root = 'drive_dataset/bbox_resized'
    os.makedirs(saved_data_root, exist_ok=True)
    for bbox_file in tqdm(os.listdir(data_root)):
        with open(os.path.join(data_root, bbox_file), 'r') as f:
            bbox_data_list = json.load(f)
        new_bbox_data_list = []
        img_w, img_h = bbox_data_list['imgWidth'], bbox_data_list['imgHeight']
        for bbox_data in bbox_data_list['objects']:
            bbox = bbox_data['2d']['modal']
            y_offset =  (img_w - correct_img_shape[1])//2
            x_offset = (img_h - correct_img_shape[0])//2
            if bbox[0] - y_offset < 0 or bbox[1] - x_offset < 0 or bbox[0] + bbox[2] - y_offset > correct_img_shape[1] or bbox[1] + bbox[3] - x_offset > correct_img_shape[0]:
                continue
            bbox_data['2d']['modal'] = [bbox[0] - y_offset, bbox[1] - x_offset, bbox[2], bbox[3]]
            new_bbox_data_list.append(bbox_data)
        bbox_data_list['objects'] = new_bbox_data_list
        with open(os.path.join(saved_data_root, bbox_file), 'w') as f:
            json.dump(bbox_data_list, f, indent=4)

def process_bbox():
    data_root = 'drive_dataset/bbox'
    saved_data_root = 'drive_dataset/bbox_processed'
    os.makedirs(saved_data_root, exist_ok=True)
    for bbox_file in tqdm(os.listdir(data_root)):
        with open(os.path.join(data_root, bbox_file), 'r') as f:
            bbox_data_list = json.load(f)
        new_bbox_data_list = {}
        for bbox_data in bbox_data_list['objects']:
            bbox = bbox_data['2d']['modal']
            label = bbox_data['label']
            if label not in new_bbox_data_list.keys():
                new_bbox_data_list[label] = []
            new_bbox_data_list[label].append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        with open(os.path.join(saved_data_root, bbox_file), 'w') as f:
            json.dump(new_bbox_data_list, f, indent=4)
            
if __name__ == "__main__":
    # preprocess_data()
    # print(correct_img_shape)
    # preprocess_bbox()
    process_bbox()