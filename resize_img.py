import cv2
import os
from tqdm import tqdm

correct_data_root = 'drive_dataset/clear'
correct_img_shape = cv2.imread(os.path.join(correct_data_root, os.listdir(correct_data_root)[0])).shape
data_root = 'drive_dataset/desnow_all'
for img in tqdm(os.listdir(data_root)):
    img_path = os.path.join(data_root, img)
    img_data = cv2.imread(img_path)
    if img_data.shape != correct_img_shape:
        img_data = cv2.resize(img_data, (correct_img_shape[1], correct_img_shape[0]), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(img_path, img_data)