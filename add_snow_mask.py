import cv2
import numpy as np
import random
import os

from tqdm import tqdm

def add_snow_mask(input_path, output_path, mask_path):
    img = cv2.imread(input_path)
    mask = cv2.imread(mask_path)
    # block_size = 100
    # block_list = []
    # for i in range(0, mask.shape[0] - block_size, block_size):
    #     for j in range(0, mask.shape[1] - block_size, block_size):
    #         block_list.append((i, j))
    if img.shape != mask.shape:
        # new_mask = np.zeros((img.shape[0] + block_size, img.shape[1] + block_size, 3), dtype=np.uint8)
        # for i in range(0, img.shape[0], block_size):
        #     for j in range(0, img.shape[1], block_size):
        #         block = random.choice(block_list)
        #         new_mask[i:i+block_size, j:j+block_size] = mask[block[0]:block[0]+block_size, block[1]:block[1]+block_size]
        new_mask = np.tile(mask, (img.shape[0] // mask.shape[0] + 1, img.shape[1] // mask.shape[1] + 1, 1))
        new_mask = new_mask[:img.shape[0], :img.shape[1], :]
        mask = new_mask
    snow = cv2.add(img, mask)
    cv2.imwrite(output_path, snow)
    
# Example usage
def main():
    clear_path = 'clear_resized'
    snow_path = 'snow_resized'
    root = 'drive_dataset'
    snow_mask_path = 'CSD/Train/Mask'
    snow_mask_list = os.listdir(snow_mask_path)
    os.makedirs(os.path.join(root, snow_path), exist_ok=True)
    for img in tqdm(os.listdir(os.path.join(root, clear_path)), ncols=100):
        if not img.endswith('.jpg') and not img.endswith('.png'):
            continue
        _ = add_snow_mask(os.path.join(root, clear_path, img), os.path.join(root, snow_path, img), os.path.join(snow_mask_path, random.choice(snow_mask_list)))

if __name__ == "__main__":
    main()