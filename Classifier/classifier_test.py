# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:10:55 2019

@author: Chen
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:20:15 2019

@author: Chen
"""

from Patch_dataset import *
from models import vgg11_bn
from models import resnet101
from torch.autograd import Variable
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import argparse


def evaluate(model,img_size,):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load data
    annotation_path = '../dataset/multi_valid_patch_annotation.csv'
    img_path = 'data/valid/'
    predict_csv = pd.read_csv(annotation_path,index_col=0)   

    print('----------loading data----------')
    dataset = patchDataset(annotation_path,img_path)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size= 1,
            shuffle=False,
            num_workers= 4,
            pin_memory=True,
        )
    print('data loaded')
    
    
    model.eval() 
        
    acc_total = 0
    acc_correct = 0 
    pre_total = 0
    pre_correct = 0
    recall_total = 0
    recall_correct = 0
    label_dict = {1:1,2:5,3:31,4:32}
    
    for batch_i, (imgs, labels,index) in enumerate(dataloader):
            
        inputs = Variable(imgs.to(device))
        labels = Variable(labels.to(device))            
        outputs = model(inputs)

        #torch.max()[0]， 只返回最大值的每个数
        #troch.max()[1]， 只返回最大值的每个索引
        predicted = torch.max(outputs.data, 1)[1]
        #print(predicted)
        
        if predicted.cpu().numpy() != 0:
            predict_csv.loc[index,'class'] = predicted.cpu().numpy()
            predict_csv.loc[index,'class'] = label_dict[int(predicted.cpu().numpy())]
            m = torch.nn.Softmax(dim=1)  # 先定义一个层。
            conf = torch.max(m(outputs.data), 1)[0]  # 使用softmax映射到0~1（其实就是sigmoid的多分类推广）
            predict_csv.loc[index,'probability'] = conf.cpu().numpy()
        else:
            predict_csv.drop(index = index,inplace = True)
        
        acc_total += labels.size(0)
        acc_correct += (predicted == labels).sum().item()
        #print(f"{batch_i}/{len(dataset)}")
        
        # 对于阳性样本中，分类的准确性。
        if predicted != 0:
            pre_total += 1
            pre_correct += (predicted == labels).sum().item()
        
        # 成功召回，且分类正确。
        if labels[0] != 0:
            recall_total += 1
            recall_correct += (predicted == labels).sum().item()
        
        
    accuracy = acc_correct/acc_total
    precision = pre_correct/pre_total
    recall = recall_correct/recall_total
    
    log_str = f"\n val_acc:{accuracy}\n val_precision:{precision}\n val_recall:{recall}"
    print(log_str)
    #predict_csv.drop(columns = 'label',inplace = True)
    #predict_csv.to_csv('../dataset/result.csv',index = False)
    return accuracy,precision,recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--model_type",type = str, default = "vgg",help="vgg/resnet")
    opt = parser.parse_args()
    
    # init model
    if opt.model_type == "vgg":
        model = vgg11_bn().to(device)
    elif opt.model_type == "resnet":
        model = resnet101(sample_size = opt.img_size, sample_duration = opt.img_size).to(device)
    if opt.weights:
        model.load_state_dict(torch.load(opt.weights))
    else:
        raise Exception("--weights is necessary!")
    print(model)
    
    evaluate(model,opt.img_size)
    