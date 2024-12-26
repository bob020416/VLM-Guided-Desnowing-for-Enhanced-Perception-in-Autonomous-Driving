import cv2
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

resized = False
desnow_all_path = 'drive_dataset/desnow_all_pred' if not resized else 'drive_dataset/desnow_all_resized_pred'
desnow_part_path = 'drive_dataset/desnow_part_pred' if not resized else 'drive_dataset/desnow_part_resized_pred'
gt_data_path = 'drive_dataset/clear_gt' if not resized else 'drive_dataset/clear_resized_gt'
saved_path = 'drive_dataset/visualization' if not resized else 'drive_dataset/visualization_resized'

os.makedirs(saved_path, exist_ok=True)

for img_path in tqdm(os.listdir(desnow_all_path)):
    desnow_all_img_path = cv2.imread(os.path.join(desnow_all_path, img_path))
    desnow_part_img_path = cv2.imread(os.path.join(desnow_part_path, img_path))
    gt_img_path = cv2.imread(os.path.join(gt_data_path, img_path))
    img = cv2.hconcat([desnow_all_img_path, desnow_part_img_path, gt_img_path])
    cv2.imwrite(os.path.join(saved_path, img_path), img)