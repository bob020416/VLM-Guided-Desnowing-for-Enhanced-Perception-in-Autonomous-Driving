import glob
import os
import random
from tqdm import tqdm
img_root = '/home/hcis-s17/multimodal_manipulation/patrick/CV/final/drive_dataset/clear'

clear_img_postfix = 'leftImg8bit.png'
bbox_postfix = 'gtBbox3d.json'
fine_postfix = 'gtFine_color.png'
img_per_folder = 100
total = 0
for folder in os.listdir(img_root):
    img_list = os.listdir(os.path.join(img_root, folder))
    random.shuffle(img_list)
    cnt = 0
    for img in tqdm(img_list, ncols=100, leave=False, desc=folder):
        image_path = os.path.join(img_root, folder, img)
        if not image_path.endswith(clear_img_postfix):
            os.remove(image_path)
            continue
        bbox_path = image_path.replace('clear', 'bbox').replace(clear_img_postfix, bbox_postfix)
        fine_path = image_path.replace('clear', 'fine').replace(clear_img_postfix, fine_postfix)
        if cnt >= img_per_folder or not os.path.exists(bbox_path) or not os.path.exists(fine_path):
            os.remove(image_path)
            if os.path.exists(bbox_path):
                os.remove(bbox_path)
            if os.path.exists(fine_path):
                for f in glob.glob(fine_path.replace(fine_postfix, '*')):
                    os.remove(f)
            continue
        cnt += 1
    total += cnt
print(total)