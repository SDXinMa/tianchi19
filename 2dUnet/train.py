# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 21:17:15 2019

@author: Chen
"""
import load_data
from model import Unet
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
# from tensorflow.keras.models import load_model
import glob
import os

'''

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
'''


class LossHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        
        global max_acc
        #if logs.get('val_acc') > savemodel_threshold and logs.get('val_acc') > max_acc:
        if logs.get('loss') <  0.1:
            nowTime1 = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M_')
            model_name = model_save_path + str(nowTime1) + str('loss={:.2f}'.format(logs.get('val_loss') * 100))
            model.save(model_name + '.h5')
            #model.save_weights(model_name + '.h5')
        max_acc = logs.get('val_acc')
'''
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        # plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
        #    plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc')
        plt.legend(loc="upper right")
        print('max valid:', max(self.val_acc['epoch']))
        global max_valid_result
        max_valid_result = max(self.val_acc['epoch'])
        plt.figure()
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.show()
'''

if __name__ == '__main__':

    train_dir = 'trainset_2d/train'
    validation_dir = 'trainset_2d/valid'
    model_save_path = 'save_models/'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
            
    valid_data_gen_args = dict(rescale=1. / 255,)
    data_gen_args = dict(rescale=1. / 255, rotation_range=1, width_shift_range=0.05, height_shift_range=0.2,
                         shear_range=0.05, zoom_range=0.05, horizontal_flip=True, fill_mode='constant', cval=0)

    input_size = (512, 512, 1)
    batch = 8
    model = Unet(input_size=input_size)
    model.summary()
    target_size = input_size[:2]    # 前两项
    train_generator = load_data.trainGenerator(batch, train_dir, data_gen_args, image_folder = 'image', mask_folder = 'label',target_size=target_size)
    valid_generator = load_data.trainGenerator(1, validation_dir, valid_data_gen_args,image_folder = 'image', mask_folder = 'label',target_size=target_size)
    history = LossHistory()
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(glob.glob('trainset_2d/train/image/'+'*.png'))//batch,
        validation_data=valid_generator,
        validation_steps=1,
        epochs=30,
        verbose = 1,
        callbacks=[history], )
    
    #history.loss_plot('epoch')
