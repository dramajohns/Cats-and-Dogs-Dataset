# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:26:37 2022

@author: fedib
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical 
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import cv2
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
def get_data_set1(filepath):
    imgs =[]
    labels=[]
    print('start reading files ...')
    for f in os.listdir(filepath):
            print('reading file: '+f)
            image = cv2.imread(filepath+f)/255
            image_array = cv2.resize(image , (256,256))
            if f.split('.')[0]=='dog':
                labels.append(0)
            else:
                labels.append(1)
            imgs.append(image_array)
            
    print('reading files finished')
    return np.array(imgs),labels  
filepath=("C:/STUDY/CIII/Deep learning/Data/")
imgs,labels = get_data_set1(filepath)
imgs.shape

def data_aug(imgs):
    data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal_and_vertical"),layers.RandomRotation(0.2),])
    augmented_imgs=[]
    for image in imgs:
        for i in range(9):
            augmented_image = data_augmentation(image)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_image[0])
            plt.axis("off")
            augmented_imgs.append(augmented_image)
    return(augmented_imgs)

augmented_imgs=data_aug(imgs)
aug_img=np.array(augmented_imgs)
aug_img.shape

newimg=aug_img.reshape(aug_img.shape[0],aug_img.shape[1],aug_img.shape[2])
#i=np.asarray(aug_img)
labels1=np.array(labels)
train_data,test_data,train_label,test_label=train_test_split(aug_img,labels1,test_size = 0.25,random_state = 0)




####################################################

Train_Data,Test_Data,Train_Label,Test_Label= train_test_split(Datanew1, Yaugmented)
## Loading vgg16 model
model1=  VGG16(weights="imagenet",include_top=False,input_shape=(200,200,3))
model1.trainable=False
#### prepocessing input
train = preprocess_input(Train_Data)
test = preprocess_input(Test_Data)

model = model.Sequential([
    model1,
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layer.Dense(32,activation='relu'),
    layer.Dense(1,activation='sigmoid')
])

