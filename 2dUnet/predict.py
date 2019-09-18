'''
model predict
'''
from model import Unet
from tensorflow.keras.models import load_model
from load_data import testGenerator
import os
import tensorflow as tf
import cv2
import numpy as np
'''
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

'''

def saveResult(save_path, test_path, npyfile,flag_multi_class = False, num_class = 2):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    images = os.listdir(test_path)
    # if num_image > len(images):
    #    num_image = len(images)
    for i, item in enumerate(npyfile):
        #print(i)
        img = item
        #img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        print(np.max(img))
        img[img > 0.4] = 255
        img[img <= 0.4] = 0
        img = img.astype(np.uint8)
        cv2.imwrite(os.path.join(save_path, "pre_"+str(images[i])), img)


if __name__ == '__main__':

 #   predict_num = 10
  #  predict_all_flag = True
    predict_data_path = 'trainset_2d/valid/image/'
    model_name = '2019-08-12_06-00_cross_loss=0.00.h5'
    
    predict_path = 'trainset_2d/valid/' + model_name
    
    images = os.listdir(predict_data_path)
#    if predict_all_flag is True or predict_num > len(images):
    predict_num = len(images)
    input_size = (512,512,1)
    testGene = testGenerator(predict_data_path, target_size=input_size[:2])

    model = load_model('save_models/' + model_name)
    model.summary()
    results = model.predict_generator(testGene, predict_num, verbose=1)
    saveResult(predict_path, predict_data_path, results)