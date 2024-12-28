import os
import cv2
import numpy as np
from tqdm import tqdm

def calculate_bbox_size(bbox):
    """
    Calculate the size of the bounding box.
    :param bbox: bounding box
    :return: size of the bounding box
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def main():
    root = './drive_dataset/snow'
    bbox_postfix = 'bbox.txt'
    image_postfix = 'leftImg8bit.png'
    total_size = 0
    cnt = 0
    for text_path in tqdm(os.listdir(root)):
        if text_path.endswith('.txt'):
            text_path = os.path.join(root, text_path)
            image_path = text_path.replace(bbox_postfix, image_postfix)
            img_shape = cv2.imread(image_path).shape[:2]
            bbox = open(text_path, 'r').read().split('\t')
            bbox = [np.clip(float(b) if 'inf' not in b else -float(b), 0, img_shape[i % 2]) for i, b in enumerate(bbox)]
            bbox = [int(b) for b in bbox]
            total_size += calculate_bbox_size(bbox)
            cnt += 1
    print('Average size of bounding box:', total_size / cnt)
    
if __name__ == '__main__':
    main()