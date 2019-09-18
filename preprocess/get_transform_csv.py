# -*- coding: utf-8 -*-
"""
Convert world annotations to pixel annotations.
@author: Chen
"""

import os
import numpy as np
import pandas as pd
import re
import SimpleITK as sitk
import glob
from tqdm import tqdm


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

def get_csv(anns_all, file_path, ):
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
    

    id_list = []
    for file in glob.glob(file_path+'*.mhd'):
        uid = re.findall(re.compile(r'\d+'),file)[0]
        id_list.append(uid)
        
        
    result = pd.DataFrame(columns = anns_all.columns)
    for index,seriesuid in enumerate(tqdm(id_list)):
        ann_df = anns_all.query('seriesuid == "%s"' % seriesuid).copy()    
        if not ann_df.empty:
            ct, origin, spacing = load_itk(file_name=seriesuid, file_path=file_path)
            # coordinate transform: world to voxel
            ann_df.coordX = (ann_df.coordX - origin[2]) / spacing[2]
            ann_df.coordY = (ann_df.coordY - origin[1]) / spacing[1]
            ann_df.coordZ = (ann_df.coordZ - origin[0]) / spacing[0]
            ann_df.diameterX = ann_df.diameterX / spacing[2]
            ann_df.diameterY = ann_df.diameterY / spacing[1]
            ann_df.diameterZ = ann_df.diameterZ / spacing[0] 
        
        #print(pd.concat([result, ann_df]))
        #print('=================')
        result = pd.concat([result, ann_df])
        #print(result)
        del ann_df
        
    #print(result)
    return result

    
file_path = '../dataset/trainset/'
test_file_path = '../dataset/testA/'
anns_path = '../dataset/chestCT_round1_annotation.csv'
anns_all = pd.read_csv(anns_path)  #读取csv文件。

v = get_csv(anns_all, file_path,)
v.to_csv('../dataset/trans_annotation.csv')