'''
Generate annotation to get patches for 3d classifier.
'''

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import pandas as pd 


def generate_annotation(model, path, conf_thres, nms_thres, img_size, batch_size, ifsave = True,save_path = '../dataset/patch_annotation.csv'):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    labels = []

    #if not os.path.exists('data/custom/predict_label'):
    #    os.makedirs('data/custom/predict_label')
    
    for batch_i, (img_path, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            #outputs = (x1, y1, x2, y2, object_conf, class_score, class_pred)
            # outputs --> predict label; target --> true label
            # to find some predicted coordinate
            # print(len(outputs))  # len(outputs) = batch_size.
            # print('img path:',img_path) # len(img_path) = batch_size
        for i,(output,path) in enumerate(zip(outputs,img_path)):
            global saving
            #save_path = path.replace('images','predict_label').replace('png','npy')
            
            # get x,y,z
            id_name = path.split('_')[0].split('/')[-1]
            z = int(path.split('_')[-1].split('.')[0])
            
            total_dict = {}
            # x1,y1,x2,y2
            total_dict['x'] = (output[:,0] + output[:,2])/2
            total_dict['y'] = (output[:,1] + output[:,3])/2
            total_dict['z'] = z
            total_dict['id'] = id_name
            
            
            if i == 0 and batch_i == 0:
                saving = pd.DataFrame(total_dict)
            else:
                saving = pd.concat([saving,pd.DataFrame(total_dict)],ignore_index=True)
            print(f'batch {batch_i} done!')


    trans_annotation_path = '../dataset/trans_annotation.csv'
    origin_data = pd.read_csv(trans_annotation_path,index_col=0)
    origin_data = origin_data[origin_data.label == 1]

    patch_data = saving
    patch_data['label'] = 0   # default --> 0 
    #print(patch_data)

    for index, row in patch_data.iterrows():
        patch_x, patch_y, patch_z, patch_id = row.x, row.y, row.z, int(row.id)
        ann = origin_data[origin_data.seriesuid == patch_id]
        for ori_index, ori_row in ann.iterrows():
            x, y, z, w, h, d = ori_row.coordX, ori_row.coordY, ori_row.coordZ, ori_row.diameterX, ori_row.diameterY, ori_row.diameterZ
            # if the predicted label is True positive
            if ((patch_x-x)/w)**2 + ((patch_y-y)/h)**2 + ((patch_z-z)/d)**2 < 1:
                patch_data.loc[index,'label'] = 1
    if ifsave:
        patch_data.to_csv(save_path)
    return patch_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/2019-08-22-11_yolov3_recall_0.87.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--dataset", type=str, default="train", help="get train or valid annotation")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    

    generate_annotation(
        model,
        path = valid_path,        
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
        save_path = '../dataset/valid_patch_annotation.csv'
    )

    generate_annotation(
        model,
        path = train_path,        
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
        save_path = '../dataset/train_patch_annotation.csv'
    )