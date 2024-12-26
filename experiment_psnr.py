import os
import cv2

from tqdm import tqdm

def psnr():
    clear_data_root = 'drive_dataset/clear'
    desnow_whole_data_root = 'drive_dataset/desnow_all'
    desnow_part_data_root = 'drive_dataset/desnow_part'
    clear_postfix = 'leftImg8bit.png'
    desnow_postfix = 'leftImg8bit.jpg'
    cnt = 0
    psnr_whole = 0
    psnr_part = 0
    for clear_img_path in tqdm(os.listdir(clear_data_root)):
        clear_img = cv2.imread(os.path.join(clear_data_root, clear_img_path))
        desnow_whole_img = cv2.imread(os.path.join(desnow_whole_data_root, clear_img_path.replace(clear_postfix, desnow_postfix)))
        desnow_part_img = cv2.imread(os.path.join(desnow_part_data_root, clear_img_path.replace(clear_postfix, desnow_postfix)))
        psnr_whole += cv2.PSNR(clear_img, desnow_whole_img)
        psnr_part += cv2.PSNR(clear_img, desnow_part_img)
        cnt += 1
    print(f'Whole PSNR: {psnr_whole / cnt}, Part PSNR: {psnr_part / cnt}')
    
if __name__ == "__main__":
    psnr()