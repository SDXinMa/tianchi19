from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
#import skimage.io as io
import os
#import skimage.transform as trans
import cv2


def trainGenerator(batch_size, train_path,  data_gen_args, image_folder = 'image', mask_folder = 'mask',image_color_mode = "grayscale",
                    mask_color_mode="grayscale", flag_multi_class=False, num_class=2,target_size=(880, 880), seed=1):

    
    print('loading data...')
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    train_dir = train_path

    image_generator = image_datagen.flow_from_directory(
        train_dir,
        classes=[image_folder],
        class_mode=None,
        batch_size =batch_size,
        target_size=target_size,
        color_mode = image_color_mode,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_dir,
        classes=[mask_folder],
        class_mode=None,
        batch_size =batch_size,
        target_size=target_size,
        color_mode = mask_color_mode,
        seed=seed)

    train_generator = zip(image_generator, mask_generator)
    for img,mask in train_generator:
        yield (img,mask)
        

def testGenerator(test_path,  target_size=(512, 512),):
    images = os.listdir(test_path)
    num_image = len(images)
    print(num_image)
    for i in range(num_image):
        img = cv2.imread(test_path+images[i], cv2.IMREAD_GRAYSCALE)
        img = img / 255
        # img = img - np.mean(img)
        img = cv2.resize(img, target_size)
        img = np.reshape(img, img.shape+(1,))
        img = np.reshape(img, (1,)+img.shape)
        yield img

