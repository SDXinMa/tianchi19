# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:42:37 2019

@author: Chen
"""

import os
import numpy as np
import pandas as pd
import re
import SimpleITK as sitk
import cv2
from tqdm import tqdm
import glob

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

def get_train_data(anns_all, file_path,data_filter=True, height=512, width=512, init_path='trainset_2d/', pics_path = 'image/',mask_path = 'label/',
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
    
    if not os.path.exists(init_path):
        #os.mkdir(init_path)
        os.makedirs(init_path+'valid/'+pics_path)
        os.makedirs(init_path+'valid/'+mask_path)
        os.makedirs(init_path+'train/'+pics_path)
        os.makedirs(init_path+'train/'+mask_path)
    id_list = []
    for file in glob.glob(file_path+'*.mhd'):
        uid = re.findall(re.compile(r'\d+'),file)[0]
        id_list.append(uid)
        
        
    for index,seriesuid in enumerate(tqdm(id_list)):
        
        if data_filter == True:
            condition = index >= len(id_list)//10
            data_dir = 'train/'
        else:
            condition = index < len(id_list)//10
            data_dir = 'valid/'
            
        if condition:
            anns_all = anns_all[anns_all.label == 1] 
            
            seriesuid = str(seriesuid)
            ann_df = anns_all.query('seriesuid == "%s"' % seriesuid).copy()   # Find Specific id
            
            ct, origin, spacing = load_itk(file_name=seriesuid, file_path=file_path)
            ct_clip = ct.clip(min=clipmin, max=clipmax)  # add windows
            del ct
            
            z_len = ct_clip.shape[0]
            for num in range(z_len):
                # only save images having annotations.ignore those without.
                img = ct_clip[num]    # (y,x)
                img = np.rint((img-np.min(img))/(np.max(img)-np.min(img))*255).astype(np.uint8)   # 归一化
                mask = np.zeros([height,width])
                
                # iterrows 返回每行的索引及一个包含行本身的对象。
                for _, ann in ann_df.iterrows():
                    # 只关注结节的标注
                    x, y, z, w, h, d = ann.coordX, ann.coordY, ann.coordZ, ann.diameterX, ann.diameterY, ann.diameterZ
                    # x,y为坐标位置，d为直径。
                    if num > z - d/2 and num < z + d / 2:
                        # find annotation images.
                        for i in range(mask.shape[0]):  
                            for j in range(mask.shape[1]):
                                if ((i-x)/w)**2 + ((j-y)/h)**2 < 1:
                                    mask[j,i] = 255 # mask --> (y,x),accoording to img
                # if mask is not a zero matrix,save it
                title = (3 - len(str(num))) * '0' + str(num)                 
                
                if data_filter == True:
                    if not np.all(mask == 0):
                        cv2.imwrite(os.path.join(init_path+data_dir+mask_path, seriesuid+'_'+title+'.png'),mask)
                        cv2.imwrite(os.path.join(init_path+data_dir+pics_path, seriesuid+'_'+title+'.png'),img)
                else:
                    cv2.imwrite(os.path.join(init_path+data_dir+mask_path, seriesuid+'_'+title+'.png'),mask)
                    cv2.imwrite(os.path.join(init_path+data_dir+pics_path, seriesuid+'_'+title+'.png'),img)    
                    
                del mask
                del img



if __name__ == '__main__':
    label_dict = {}
    label_dict[1]  = 'nodule'
    label_dict[5]  = 'stripe'
    label_dict[31] = 'artery'
    label_dict[32] = 'lymph'
    
    file_path = '../dataset/trainset/'
    test_file_path = '../dataset/testA/'
    anns_path = '../dataset/trans_annotation.csv'  # 直接从这里读更爽啊！
    anns_all = pd.read_csv(anns_path)  #读取csv文件。
    
    # get mask and pics
    get_train_data(anns_all, file_path,init_path='trainset_2d/',data_filter = True)
    
    # get valid
    get_train_data(anns_all, file_path,init_path='trainset_2d/',data_filter = False)