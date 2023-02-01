# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:21:40 2022

@author: fedib
"""
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical 
from matplotlib import pyplot as plt
def get_data_set(filepath):
    imgs =[]
    labels=[]
    print('start reading files ...')
    for f in os.listdir(filepath):
        if not (f.endswith('pgm')):
            labels.append(f.split('.')[0])
            print('reading file: '+f)
            img = np.asarray(Image.open(filepath+f))
            imgs.append(img)
    print('reading files finished')
    return np.asarray(imgs),labels       

filepath=("C:/STUDY/CIII/Deep learning/tp/tp3/yalefaces/")
imgs,labels = get_data_set(filepath)
labels1=[i[-2:] for i in labels]    
labels1=np.asarray(labels1)

train_data,test_data,train_label,test_label=train_test_split(imgs,labels1,test_size = 0.33,random_state = 0)
train_data1=train_data.reshape(train_data.shape[0],train_data.shape[1],train_data.shape[2],1)
test_data1=test_data.reshape(test_data.shape[0],test_data.shape[1],test_data.shape[2],1)

train_label1= to_categorical(train_label)
test_label1= to_categorical(test_label)

train_label2= train_label1[:,1:]
test_label2= test_label1[:,1:]

classifier= Sequential()
classifier.add(Convolution2D(32, kernel_size=9,input_shape=(243,320,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(64, kernel_size=9, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(128, kernel_size=9, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())
classifier.add(Dense(512,activation='relu'))
classifier.add(Dense(15,activation='softmax'))
classifier.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])

result=classifier.fit(train_data1,train_label2,batch_size=20,epochs=10,validation_data=(test_data1,test_label2))    
classifier.summary()


plot_folder = "plot"
plt.plot(result.history['accuracy'], label='accuracy')
plt.plot(result.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.1, 1])
plt.legend(loc='lower right')


def get_random_img():
    import random
    position=["centerlight","glasses","happy","leftlight","noglasses","normal","rightlight","sad","sleepy","surprised","wink"]
    random_class=random.randint(1, 15)
    random_position=random.choice(position)
    if random_class<10:
        img_path = f"C:/STUDY/CIII/Deep learning/tp/tp3/yalefaces/subject0{random_class}.{random_position}"
    else :
        img_path = f"C:/STUDY/CIII/Deep learning/tp/tp3/yalefaces/subject{random_class}.{random_position}"
    
    img = np.asarray(Image.open(img_path))
    return img, random_class
    

get_random_img()
img, expected_class = get_random_img()
img1=img.reshape(img.shape[0],img.shape[1],1)
scaled_img = np.expand_dims(img1, axis=0)
results = classifier.predict(scaled_img)
result = np.argmax(results, axis=1)
index = result[0]
print(expected_class,index+1)