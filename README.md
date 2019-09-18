# Introduction
- This repository is an implement of lung detection and classification.The data is from a competetion from the following link: https://tianchi.aliyun.com/competition/entrance/231724/information
- The model was built on **Pytorch 1.1.0**

# File Description
### preprocess
- `.ipynb` are used for visualization and test.I don't arrange them well.
- `.py` are runnable code.

### dataset
- Download from https://tianchi.aliyun.com/competition/entrance/231724/information
- Including `testA`, `trainset`,`chestCT_round1_annotation.csv`

### yolo
- Main code from the following git, I just do some modification to ensure the code work well in my dataset.
https://github.com/eriklindernoren/PyTorch-YOLOv3

### 3DCNN 
- Use 3D Classification Network to make a futher classification and reduce false positive.
- Mainly use 3D version's VGG and Resnet to do the classification.

# How to use

### preprocess
1. Run `get_transform_csv.py`
2. Run `kmeans_for_anchors.py` to get anchors and modify the NetWork config `yolov3-custom.cfg`.

### YOLOv3 for 2D images detection
1. Run `get_data.py` to convert `mhd` files to `png` files, and label for them.
2. Run `train.py` to train yolo model, there are many argument you can set.I change the default setting to make it more convenient. You may care about the epochs, you can refer to `yolo.sh`
3. Run `test.py` to evaluate the effect of the model on valid data.**Make sure the argument --weights_path in a right path**
4. Run `detect.py` to generate annotation image to dir `output`.(not necessary,just visualize to check the result)**Make sure the argument --weights_path is a right path**

### dir Classifier for 3D patchs classification
1. Run `yolo/gen_annotation.py` to get some predicted annotation, saving in `dataset/train_patch_annotation.csv` and `dataset/valid_patch_annotation.csv`
2. Run `Patch_dataset.py` to convert mhd to npy,which can impore the training speed.(reading npy is much faster than mhd, Exchange space for time). (change params `nms_thres` `conf_thre` can generate different samples,to balance True and Flase samples,I set both 0.5)
3. Run `classifier_train.py` to train the model.
4. Run `classifier_test.py --weights ` to test the model.