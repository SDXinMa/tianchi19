# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 00:39:12 2019

@author: Chen
"""

from __future__ import division


from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter
import sys
sys.path.append('../eval/')
from evaluation import *
from gen_annotation import *

#import torch.distributed as dist
#from torch.multiprocessing import Process
# from torchsummay import summary

if __name__ == "__main__":

    #torch.distributed.init_process_group(backend='nccl',init_method='tcp://127.0.0.1:23456',world_size=4,rank=0)
    #print(torch.distributed.is_nccl_available())

    max_recall = 0
    loss_list = []
    val_recall_list = []
    froc_list = []
    map_list = []

        # save in logs/
    writer = SummaryWriter('logs/')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=5, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)
    print(model)
    
    # 分发模型
    #model = torch.nn.parallel.DistributedDataParallel(model)
    #print('model distributed')

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)
    '''
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(opt.img_size),
        transforms.RandomRotation(10),
    ])
    '''

    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    # 分发数据
    #train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
   # print('data distributed')
    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        #train_sampler.set_epoch(epoch)
        model.train()
        start_time = time.time()
        for batch_i, (paths, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i
            
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            

            loss, outputs = model(imgs, targets)
            loss.backward()
            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()
                # ----------------
                #   Log progress
                # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

                # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]
                    
            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            #model.seen += imgs.size(0)
            print(f"\nTotal loss {loss.item()}")
        print(log_str)                
        writer.add_scalar('loss',loss.item(),epoch)
        loss_list.append(loss.item())

        
        if (epoch+1) % opt.evaluation_interval == 0:
            print("\n------- Evaluating Model -------")
            # Evaluate the model on the validation set

            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=1,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            
            # compute FROC:
            generate_annotation(
                model,
                path =  'data/custom/predict.txt',        
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
                save_path = '../dataset/tmp_result.csv'
            )

            annotations_filename = '../dataset/trans_annotation.csv'
            results_filename = '../dataset/tmp_result.csv'
            outputDir = '../eval/myresult'
            score = noduleCADEvaluation(annotations_filename,results_filename,outputDir)
            froc_list.append(score)
            print(f'---- score {score}')
            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP",]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            #print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            print(f"---- val_recall {recall.mean()}")
            val_recall_list.append(recall.mean())
            
        # find the best recall model !!!
        if (epoch+1) % opt.checkpoint_interval == 0:
            now_date = datetime.datetime.now().strftime('%m-%d-%H')
            torch.save(model.state_dict(), f"checkpoints/{now_date}_froc_{score:.2f}_recall_{recall.mean():.2f}%d.pth" % (epoch+1))
    
    plt.plot(loss_list)
    plt.savefig('logs/loss.png')
    plt.plot(val_recall_list)
    plt.savefig('logs/val_recall_list.png') 
    plt.plot(froc_list)
    plt.savefig('logs/froc.png') 