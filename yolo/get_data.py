"""
Created on Thu Aug  8 15:42:37 2019
1. 将3D图像转化为2D图像存放在文件夹中（所有图片都要进行的操作）
2. 生成标注，有标注数据的图片才会生成标注文件。
3. 1:19 划分验证集和训练集，保存为train.txt,valid.txt; 
保存最终预测用的图片在predict.txt（相比valid来说，包括了那些没有标注的2D图片）
@author: Chen
"""

import os
import numpy as np
import pandas as pd
import re
import SimpleITK as sitk
import cv2
from tqdm import tqdm
from glob import glob
import shutil


def load_itk(file_name, file_path):
    '''
    modified from https://stackoverflow.com/questions/37290631/reading-mhd-raw-format-in-python
    '''
    
    # Reads the image using SimpleITK
    file = os.path.join(file_path, file_name + '.mhd')
    itkimage = sitk.ReadImage(file)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.  
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension   (z,y,x)
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing

def get_train_valid_data(anns_all, file_path, height=512, width=512, init_path='data/custom/', pics_path = 'images/',mask_path = 'labels/',
              clipmin= -1000, clipmax=600):
    '''
    get mask and save
    '''

    '''
    input:
    seriesuid: specify the scan plotted.
    anns_all:  the annotation provided (Dataframe).
    file_path: the path of the data.
    plot_path: the path of the visualization, default: make a subdirectory under the current dir.   
    clip_min:  the lower boundary which is used for clipping the CT valued for the lung window.
    clip_max:  the upper boundary which is used for clipping the CT valued for the lung window.
    only_df:   if True, only return the dataframe , and do not plot.
    return_ct: if True, return the dataframe with the ct array.
    
    return:
    ann_df:    return the annotation dataframe according to the seriesuid
    
    Mediastinum window: clipmin=-150, clipmax=250
    '''


    
    if os.path.exists(init_path+pics_path) and os.path.exists(init_path+mask_path):
        shutil.rmtree(init_path+pics_path)
        shutil.rmtree(init_path+mask_path)
    
    if not os.path.exists(init_path+pics_path):
        #os.mkdir(init_path)
        os.makedirs(init_path+pics_path)
        
    if not os.path.exists(init_path+mask_path):
        os.makedirs(init_path+mask_path)
    id_list = []
    for file in glob(file_path+'*.mhd'):
        uid = re.findall(re.compile(r'\d+'),file)[0]
        id_list.append(uid)

    classes = ['nodule','stripe','artery','lymph']    
    
    valid_id = []
    for index,seriesuid in enumerate(id_list):
        if index < len(id_list)//20:
            valid_id.append(seriesuid)   # save valid id in a list
    
        seriesuid = str(seriesuid)
        ann_df = anns_all.query('seriesuid == "%s"' % seriesuid).copy()   # Find Specific id
        ct, origin, spacing = load_itk(file_name=seriesuid, file_path=file_path)
        ct_clip = ct.clip(min=clipmin, max=clipmax)  # add windows
        del ct
        z_len = ct_clip.shape[0]
        print(f'{index+1}/{len(id_list)}done!')
        
        for num in range(z_len):
            img = ct_clip[num]
            img = np.rint((img-np.min(img))/(np.max(img)-np.min(img))*255).astype(np.uint8)   # 归一化            
            # iterrows 返回每行的索引及一个包含行本身的对象。            
            title = (3 - len(str(num))) * '0' + str(num)
            # save all 2d images.    
            cv2.imwrite(os.path.join(init_path+pics_path, seriesuid+'_'+title+'.png'),img)
            for _, ann in ann_df.iterrows():
                # classes --> all classes ['nodule','stripe','artery','lymph'] 
                if label_dict[ann.label] in classes:
                    # x,y为坐标位置，d为直径。
                    x, y, z, w, h, d = ann.coordX, ann.coordY, ann.coordZ, ann.diameterX, ann.diameterY, ann.diameterZ
                    if num > z - d/2 and num < z + d / 2:
                        txt_path = os.path.join(init_path+mask_path, seriesuid+'_'+title+'.txt')
                        if classes.index(label_dict[ann.label]) in [0,1,2,3]:
                            if ann.coordX/512<1 and ann.coordY/512 <1 and ann.diameterX/512 < 1 and ann.diameterX/512 <1 and ann.diameterY/512 < 1:
                                # 官方的标注有问题！！！剔除掉不符合的条件的！！！（383281）
                                with open(txt_path,'a') as f:
                                    f.write(f'{classes.index(label_dict[ann.label])} '+str(ann.coordX/512)+' '+str(ann.coordY/512)+' '+str(ann.diameterX/512)+' '+str(ann.diameterY/512))
                                    f.write('\n')
                            
    define_train_valid(valid_id)
        

def define_train_valid(valid_id,img_path = 'data/custom/images/',txt_path = 'data/custom/labels/',):
    img_list = glob(img_path+'*.png')
    label_list = glob(txt_path + '*.txt')
    print('2D img length:',len(img_list))
    print('label length:',len(label_list))
    valid_file = open('data/custom/valid.txt','w') 
    train_file = open('data/custom/train.txt','w')
    predict_file = open('data/custom/predict.txt','w')

    print('valid id length:',len(valid_id))
    for i in range(len(label_list)): 
        if re.findall('\d+',label_list[i])[0] in valid_id:
            valid_file.write(label_list[i].replace('labels','images').replace('txt','png'))
            valid_file.write('\n')       
        else:
            train_file.write(label_list[i].replace('labels','images').replace('txt','png'))
            train_file.write('\n')

    for i in range(len(img_list)):
        if re.findall('\d+',img_list[i])[0] in valid_id:
            predict_file.write(img_list[i])
            predict_file.write('\n')
    train_file.close()
    valid_file.close()
    predict_file.close()


if __name__ == '__main__':
    label_dict = {}
    label_dict[1]  = 'nodule'
    label_dict[5]  = 'stripe'
    label_dict[31] = 'artery'
    label_dict[32] = 'lymph'
    
    file_path = '../dataset/trainset/'
    test_file_path = '../dataset/testA/'
    anns_path = '../dataset/trans_annotation.csv'  # transformed
    anns_all = pd.read_csv(anns_path, index_col=0)  #读取csv文件。
    # get LABEL 
    get_train_valid_data(anns_all, file_path,init_path='data/custom/', pics_path = 'images/',mask_path = 'labels/')
