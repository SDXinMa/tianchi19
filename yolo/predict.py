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
import pandas as pd
from gen_annotation import generate_annotation
from Classifier.Patch_dataset import DatasetGenerator
from tqdm import tqdm


def generate_2d(file_path, id_list, clipmin= -1000, clipmax=600):
    # get 2D images from mhd ID list.

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--yolo_weights_path", type=str, default = 'checkpoints/2019-09-04-23_recall_0.9467455621301775_ckpt_19.pth',help="if specified starts from checkpoint model")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--model_type",type = str, default = "vgg",help="vgg/resnet")
    parser.add_argument("--classifier_weights_path",default="../Classifier/checkpoints/vgg_0.9926158879365957_47.pth",type=str, help="if specified starts from checkpoint model")
    opt = parser.parse_args()
    print(opt)     

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #---------------------------------detect-----------------------------
    detect_model = Darknet(opt.model_def, img_size=opt.img_size).to(device)    
    detect_model.load_state_dict(torch.load(opt.yolo_weights_path))
    predict_path = 'data/custom/predict.txt'
    img_path = '../dataset/trainset/'
    tmp_path = '../dataset/multi_predict_patch_annotation.csv'
    
    predict_csv = generate_annotation(
        detect_model,
        path = predict_path,        
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
        save_path = tmp_path,
    )
    
    # -------------------------false positive reduction------------------------
    if opt.model_type == "vgg":
        classify_model = vgg11_bn().to(device)
    elif opt.model_type == "resnet":
        classify_model = resnet101(sample_size = opt.img_size, sample_duration = opt.img_size).to(device)
    if opt.classifier_weights_path:
        classify_model.load_state_dict(torch.load(opt.classifier_weights_path))
    else:
        raise Exception("--classifier_weights_path is necessary!")
    

    dataset = DatasetGenerator(tmp_path,img_path)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size= 1,
            shuffle=False,
            num_workers= 4,
            pin_memory=True,
        )
    
    print('data loaded')
    classify_model.eval() 

    label_dict = {1:1,2:5,3:31,4:32}    
    for batch_i, (imgs, labels,index) in enumerate(dataloader):
        #print(imgs.size())
        imgs = imgs.unsqueeze(0).float()
        inputs = Variable(imgs.to(device))
        labels = Variable(labels.to(device))            
        outputs = classify_model(inputs)

        #torch.max()[0]， 只返回最大值的每个数
        #troch.max()[1]， 只返回最大值的每个索引
        predicted = torch.max(outputs.data, 1)[1]
        if predicted.cpu().numpy() != 0:
            predict_csv.loc[index,'class'] = predicted.cpu().numpy()
            predict_csv.loc[index,'class'] = label_dict[int(predicted.cpu().numpy())]
            m = torch.nn.Softmax(dim=1)  # 先定义一个层。
            conf = torch.max(m(outputs.data), 1)[0]  # 使用softmax映射到0~1（其实就是sigmoid的多分类推广）
            predict_csv.loc[index,'probability'] = conf.cpu().numpy()
        else:
            predict_csv.drop(index = index,inplace = True)
    predict_csv.drop(columns = 'label',inplace = True)
    predict_csv.to_csv('../dataset/result.csv',index = False)



    #print(detect_model)
    '''
    print("------------Beginning------------")
    detect_model.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # only get valid mhd
    predict_path = 'data/custom/valid.txt'
    with open(predict_path,'r') as f:
        valid_ids = f.readlines()
        valid_ids = set(map(lambda x: re.findall("\d+",x)[0],valid_ids))  # str id 
    id_list = list(valid_ids)
    
    file_path = '../dataset/trainset/'
    print(id_list)
    print(f'num of images:{len(id_list)}')
    


    
    # detect
    saving = pd.DataFrame({})
    for (img,img_id,z) in generate_2d(file_path,id_list):
        img = Variable(img.type(Tensor), requires_grad=False)
        with torch.no_grad():
            outputs = detect_model(img)
            outputs = non_max_suppression(outputs, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)
            # outputs --> x1, y1, x2, y2, object_conf, class_score, class_pred
            # print(outputs.size())
            for i,output in enumerate(outputs):
                #if outputs[0] is not None:
                #    print(output.size())
                #    print(f'id:{img_id},z:{z},coord:{(output[:,0] + output[:,2])/2},{(output[:,1] + output[:,3])/2}')
                total_dict = {}     
                # x1,y1,x2,y2
                if output is not None:
                    total_dict['seriesuid'] = img_id
                    total_dict['coordX'] = (output[:,0] + output[:,2])/2
                    total_dict['coordY'] = (output[:,1] + output[:,3])/2
                    total_dict['coordZ'] = z

                    saving = pd.concat([saving,pd.DataFrame(total_dict)],ignore_index=True)
    print(saving)
    #for i in id_list:
        #generate_patch()
    '''
    
    