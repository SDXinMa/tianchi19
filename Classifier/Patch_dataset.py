# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:03:31 2019

@author: Chen
"""


from torch.utils.data import Dataset
import pandas as pd
from glob import glob
import re
import torch
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=True):
    '''
    used for resample
    '''
    
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    # x,y,z
    out_size = [int(np.round(original_size[0]*(original_spacing[0]/out_spacing[0]))),
                int(np.round(original_size[1]*(original_spacing[1]/out_spacing[1]))),
                int(np.round(original_size[2]*(original_spacing[2]/out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image),original_spacing,original_size



def load_itk(file_name, file_path,):
    '''
    modified from https://stackoverflow.com/questions/37290631/reading-mhd-raw-format-in-python
    '''
    
    # Reads the image using SimpleITK
    file = os.path.join(file_path, file_name + '.mhd')
    itkimage = sitk.ReadImage(file)
    itkimage,original_spacing,original_size = resample_image(itkimage)
    # Read the spacing along each dimension   (z,y,x)
    spacing = np.array(list(reversed(itkimage.GetSpacing())))   # origin spcacing, resample spacing =>>1,1,1.
    
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.  
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    return ct_scan, origin, original_spacing,original_size


class DatasetGenerator(Dataset):
    def __init__(self,trans_annotation_path,img_path,patch_size=(32,32,32),clipmin= -1000, clipmax=600):
        super(DatasetGenerator).__init__()
        ann_all = pd.read_csv(trans_annotation_path)
        # nodules' coordinates.
        self.ids,self.xs,self.ys,self.zs,self.labels = \
            ann_all.id.values,ann_all.x.values,ann_all.y.values,ann_all.z.values,ann_all.label.values
            
        self.img_path = img_path
        self.patch_size = patch_size  # z,y,x
        self.clip_range = [clipmin,clipmax]
        
    def __getitem__(self,index):
        
        seriesuid = str(self.ids[index])
        #print(seriesuid)
        
        # get patch by id
        ct, origin, original_spacing,original_size = load_itk(file_name=seriesuid, file_path=self.img_path)
        ct_clip = ct.clip(min=self.clip_range[0], max=self.clip_range[1])  # lung windows, ct-->z,y,x
        #print(self.xs[index],self.ys[index],self.zs[index])
        
        # get real(world) coord/diameter , space-->1*1*1(mm)
        x =  int(np.round(self.xs[index] * (original_spacing[0])))
        y =  int(np.round(self.ys[index] * (original_spacing[1])))
        z =  int(np.round(self.zs[index] * (original_spacing[2])))
        
        cube_size = self.patch_size
        n = cube_size[2]//2
        # pad to make sure patches not empty 
        ct_clip = np.pad(ct_clip,n,'constant')
        x,y,z = x+n,y+n,z+n  # fixed site
        img = ct_clip
        img = np.rint((img-np.min(img))/(np.max(img)-np.min(img))*255).astype(np.uint8)   # 归一化
        img = img[z-cube_size[0]//2:z+cube_size[0]//2,y-cube_size[1]//2:y+cube_size[1]//2,x-cube_size[2]//2:x+cube_size[2]//2]  # get patches(64,64,64)
        
        return img, self.labels[index],index
    
    
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.labels)


class patchDataset(Dataset):
    def __init__(self,trans_annotation_path,img_path):
        super(patchDataset).__init__()   
        ann_all = pd.read_csv(trans_annotation_path)    
        # nodules' coordinates.
        self.labels = ann_all.label.values
        self.img_path = img_path


    def __getitem__(self,index):
        label = torch.from_numpy(np.array(self.labels[index])) # 1 --> nodule
        npy_path = os.path.join(self.img_path,str(index)+'.npy')
        img = np.load(npy_path)
        img = torch.from_numpy(img).unsqueeze(0).float()
        return img,label,index

        
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.labels)


if __name__ == '__main__':
    annotation_path = '../dataset/train_patch_annotation.csv'
    valid_annotation_path = '../dataset/valid_patch_annotation.csv'
    img_path = '../dataset/trainset/'
    generateDataset = DatasetGenerator(annotation_path,img_path)
    generateValid = DatasetGenerator(valid_annotation_path,img_path)

    if not os.path.exists("data/train"):
        os.makedirs("data/train")
    if not os.path.exists("data/valid"):
        os.makedirs("data/valid")
    
    for i,(img,label,index) in enumerate(tqdm(generateDataset)):
        # saving data with the name of index.
        np.save(os.path.join("data/train",str(index)+'.npy'), img)
        

    # generate valid dataset.
    for i,(img,label,index) in enumerate(tqdm(generateValid)):
        # saving data with the name of index.
        np.save(os.path.join("data/valid",str(index)+'.npy'), img)


# for test
    dataset = patchDataset(annotation_path,'data/train')
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=5,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    
    for batch_i, (imgs, labels) in enumerate(dataloader):
        print(imgs.dtype)
        print(imgs)
        print(imgs.shape)
        print(labels)
        break
        
        
    '''   
        # visualization
        img = imgs[0][0]
        plt.figure(figsize=(36,36))
        for i in range(img.shape[0]):
            plt.subplot(6,6,i+1)
            plt.imshow(img[i],cmap = plt.cm.gray)
        
        break
    '''