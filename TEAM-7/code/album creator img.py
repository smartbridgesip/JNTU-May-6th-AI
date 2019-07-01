# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:31:38 2019

@author: malle
"""

import numpy as np
from keras.preprocessing import image

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
from keras.models import load_model
classifier = load_model(r'C:\Users\malle\intelligent album creator.h5')
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)
import cv2

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img():
    x = openfn()
    test_image = image.load_img(x, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    img1 = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict_classes(test_image)
    print(result)
    index=['Manas','madhavi','swathi']	
    label = Label( root, text="Prediction : "+str(index[result[0]]))
    label.pack()
    img = Image.open(x)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()
    FaceFileName="file_"+str(np.random.rand())+'.jpg'
    if (result[0]==0):
        cv2.imwrite("C:\\Users\\malle\\Desktop\\intelligent album creator\\\\Manas\\"+FaceFileName, img1)
    elif (result[0]==1):
        cv2.imwrite(r"C:\\Users\\malle\\Desktop\\intelligent album creator\\Model responce\\madhavi\\"+FaceFileName, img1)
    elif (result[0]==2):
        cv2.imwrite(r"C:\\Users\\malle\\Desktop\\intelligent album creator\\swathi\\"+FaceFileName, img1)

btn = Button(root, text='open image', command=open_img).pack()

root.mainloop()

