# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:20:15 2019

@author: Chen
"""

from Patch_dataset import *
from models import vgg11_bn
from models import *
from torch.autograd import Variable
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
import time
import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=50, help="size of each image batch")
    parser.add_argument("--pretrained_weights",type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--model_type",type = str, default = "vgg",help="vgg/resnet")
    opt = parser.parse_args()


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # init model
    if opt.model_type == "vgg":
        model = vgg11_bn().to(device)
    elif opt.model_type == "resnet":
        model = resnet101(sample_size = opt.img_size, sample_duration = opt.img_size).to(device)
    if opt.pretrained_weights:
        model.load_state_dict(torch.load(opt.pretrained_weights))

    print(model)
    
    
    # load data
    annotation_path = '../dataset/train_patch_annotation.csv'
    img_path = 'data/train/'
    
    print('----------loading data----------')
    dataset = patchDataset(annotation_path,img_path)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size= opt.batch_size,
            shuffle=True,
            num_workers= 4,
            pin_memory=True,
        )
    print('data loaded')
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()  
    max_acc = 0
    
    for epoch in range(opt.epochs):
        model.train() 
        start_time = time.time()
        
        total = 0
        correct = 0 
        
        # start training.
        for batch_i, (imgs, labels) in enumerate(dataloader):
            
            inputs = Variable(imgs.to(device),requires_grad=True)
            labels = Variable(labels.to(device),requires_grad=False)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()   
            
            predicted = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            accuracy = correct/total
            # log
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch + 1, opt.epochs, batch_i, len(dataloader))
            log_str += f"\n loss:{loss.item()}"
            log_str += f"\n acc:{accuracy}"
            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"
            print(log_str)            
        
        
        if accuracy>max_acc and accuracy > 0.9:
            max_acc = accuracy
            torch.save(model.state_dict(), f"checkpoints/{opt.model_type}_{max_acc}_%d.pth" % epoch)
            