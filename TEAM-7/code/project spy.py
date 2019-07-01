# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:53:47 2019

@author: malle
"""

import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
model_cnn=Sequential()
model_cnn.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(2,2)))
model_cnn.add(Flatten())
model_cnn.add(Dense(100,activation='relu'))
model_cnn.add(Dense(3,activation='softmax'))
model_cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
x_train=train_datagen.flow_from_directory(r"C:\Users\malle\Desktop\album creator\training",target_size=(64,64),batch_size=32,class_mode='categorical')
x_test=test_datagen.flow_from_directory(r"C:\Users\malle\Desktop\album creator\testing",target_size=(64,64),batch_size=32,class_mode='categorical')
print(x_train.class_indices)
model_cnn.fit_generator(x_train,samples_per_epoch=3600,epochs=100,validation_data=x_test,nb_val_samples=900)
model_cnn.save("intelligent album creator.h5")