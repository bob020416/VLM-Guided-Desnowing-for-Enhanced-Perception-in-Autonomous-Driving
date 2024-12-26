import cv2
import numpy as np
import os
import sys
from keras.models import load_model
from argparse import ArgumentParser
import time
import tensorflow as tf
import keras.backend as K

from keras.preprocessing.image import ImageDataGenerator

#custom
import model.model 

print('import end')

def parse_args():
    parser = ArgumentParser(description='Predict')
    parser.add_argument(
        '-i', '--dataroot',
        type=str, default='./testImg',
        help='root of the image, if data type is npy, set datatype as npy'
    )
    parser.add_argument(
        '-dtype', '--datatype',
        type=str, default=['jpg','tif','png'],
        help='type of the image, if == npy, will load dataroot'
    )
    parser.add_argument(
        '-o', '--predictpath',
        type=str, default='./predictImg',
        help='root of the output'
    )
    parser.add_argument(
        '-b_size', '--batch_size',
        type=int, default=3,
        help='batch_size'
    )
    parser.add_argument(
        '-c', '--crop', default=False,
        action='store_true',
    )
    
    return  parser.parse_args()

def progress(count, total, status=''):
    bar_len = 20
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '|' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    if count != total:
        sys.stdout.flush()
    else:
        print()
    
    
    
def generate_data_generator(datagenerator, X,BATCHSIZE):
    genX1 = datagenerator.flow(X,batch_size = BATCHSIZE,shuffle=False)
    count = 0
    while True:
            Xi1 = genX1.next()
            
            Xi1 = Xi1/255
            yield [Xi1]
            
if __name__== '__main__':
    
    args = parse_args()
    ram_limit_img_size = 100
    total_time = 0
    
    #read test data
    selectNames = []
    data=[]
    if args.crop:
        bbox_list = []
    print('Read img from:' , args.dataroot)
    fnames=os.listdir(args.dataroot)
    print('Len of the file:',len(fnames))
    count = 0

    print('----------Build Model----------')
    model=model.model.build_DTCWT_model((512,672,3))
    model.load_weights('./modelParam/finalmodel.h5',by_name=False)

    print('LogPath:',args.predictpath)

    val_data_gen = ImageDataGenerator(featurewise_center=False,
                        featurewise_std_normalization=False)
    for idx, f in enumerate(fnames):
        progress(count,min(ram_limit_img_size, len(fnames)),'Loading data...')
        if f.split('.')[-1] in args.datatype:
            tmp=cv2.imread(args.dataroot+'/'+f)
            if args.crop:
                bbox_file = os.path.join(args.dataroot, f.replace('leftImg8bit.png', 'bbox.txt'))
                if not os.path.exists(bbox_file):
                    continue
                bbox = open(bbox_file, 'r').read().split('\t')
                bbox = [np.clip(float(b) if 'inf' not in b else -float(b), 0, tmp.shape[i % 2]) for i, b in enumerate(bbox)]
                bbox = [int(b) for b in bbox]
                bbox_list.append(bbox)
                tmp = tmp[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                # cv2.imwrite('test.png', tmp)
            selectNames.append(f)
            if tmp.shape[1]<tmp.shape[0]:
                tmp=np.rot90(tmp)
            if tmp.shape[0]!=480 or tmp.shape[1]!=640:
                tmp=cv2.resize(tmp, (640, 480), interpolation=cv2.INTER_CUBIC)
            data.append(tmp)
            count+=1
        else:
            continue
        if count == ram_limit_img_size or idx == len(fnames)-1:
            data=np.array(data)
            print(data.shape,'data shape')
            print('Start Padding')
            for i in range(data.shape[0]):
                progress(i+1,data.shape[0],'Paddding and convert data to YCRCB...')
                data[i]=cv2.cvtColor(data[i],cv2.COLOR_BGR2YCR_CB)
            data=np.pad(data,((0,0),(16,16),(16,16),(0,0)),'constant')
            print(data.shape,'data shape')
            #data=data/255

            if not os.path.exists(args.predictpath):
                os.mkdir(args.predictpath)
            
            #BUILD COMBINE MODEL
            
            start_time = time.time()
            pred=model.predict_generator(generate_data_generator(val_data_gen,data,args.batch_size),steps = data.shape[0]/args.batch_size,verbose=1)
            total_time += time.time() - start_time
            
            print('Save Output')
            for i in range(pred.shape[0]):
                progress(i+1,pred.shape[0],'Saving output...')
                pred[i]=np.clip(pred[i],0.0,1.0)
                if args.crop:
                    original_img = cv2.imread(os.path.join(args.dataroot, selectNames[i]))
                    original_img[bbox_list[i][0]:bbox_list[i][2], bbox_list[i][1]:bbox_list[i][3]] = cv2.resize(cv2.cvtColor( (pred[i]*255).astype(np.uint8), cv2.COLOR_YCrCb2BGR), (bbox_list[i][3] - bbox_list[i][1], bbox_list[i][2] - bbox_list[i][0]))
                    cv2.imwrite(args.predictpath+'/'+os.path.splitext(selectNames[i])[0]+'.jpg', original_img)
                else:
                    cv2.imwrite(args.predictpath+'/'+os.path.splitext(selectNames[i])[0]+'.jpg',cv2.cvtColor( (pred[i]*255).astype(np.uint8), cv2.COLOR_YCrCb2BGR))
            data = []
            selectNames = []
            if args.crop:
                bbox_list = []
                
            count = 0
    print(f'Total time: {total_time}s')