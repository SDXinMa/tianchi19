'''
1.get 2d slice from mhd
2.using yolo to get candidates
3.using candidates' coordinate to get 32*32*32 patchs
4.using 3D CNN to reduce False positive.
5.get result.
'''

import argparse
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from Classifier.models import vgg11_bn
from models import *
from utils.utils import *
from get_data import load_itk
import re
from glob import glob
import torch

def generate_2d(file_path, clipmin= -1000, clipmax=600):

    # only get valid mhd
    with open('data/custom/valid.txt','r') as f:
        valid_ids = f.readlines()
        valid_ids = list(map(lambda x: re.findall("\d+",x)[0],valid_ids))  # str id 
    id_list = []
    for file in glob(file_path+'*.mhd'):
        uid = re.findall(re.compile(r'\d+'),file)[0]
        if uid in valid_ids:
            id_list.append(uid)
    print(f'num of images:{len(id_list)}')

    for index,seriesuid in enumerate(id_list):
        seriesuid = str(seriesuid)
        ct, origin, spacing = load_itk(file_name=seriesuid, file_path=file_path)
        ct_clip = ct.clip(min=clipmin, max=clipmax)  # add windows
        del ct
        z_len = ct_clip.shape[0]
        for num in range(z_len):
            img = ct_clip[num]
            img = np.rint((img-np.min(img))/(np.max(img)-np.min(img))*255).astype(np.uint8)   # 归一化            
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
            yield img, seriesuid, num


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--yolo_weights_path", type=str, default = 'checkpoints/2019-08-22-11_yolov3_recall_0.87.pth',help="if specified starts from checkpoint model")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")

    opt = parser.parse_args()
    print(opt)     
    
    classify_model = vgg11_bn()
    detect_model = Darknet(opt.model_def, img_size=opt.img_size).to(device)    
    detect_model.load_darknet_weights(opt.yolo_weights_path)
    print(detect_model)

    print("------------Beginning------------")
    detect_model.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    file_path = '../dataset/trainset/'

    # detect
    for (img,img_id,z) in generate_2d(file_path):
        img = Variable(img.type(Tensor), requires_grad=False)
        plt.show()
        with torch.no_grad():
            outputs = detect_model(img)
            outputs = non_max_suppression(outputs, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)
            if outputs!= None:
                print(outputs)

            
        
        