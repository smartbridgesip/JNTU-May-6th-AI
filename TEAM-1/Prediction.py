import numpy as np
from keras.preprocessing import image

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
val = ''
from keras.models import load_model
classifier = load_model('smart security systems using image recoginition.h5')
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img():
    x = openfn()
    test_image = image.load_img(x, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict_classes(test_image)
    print(result)
    index=['manasa','swapna','unknow']
    val=''
    if((result == [0]) | (result == [1])):
        val = "Authorized Person"
        print("Authorized")
    else:
        val = "UN Authorized Person"
        print(" Un Authorized")
        
    label = Label( root, text="Prediction : "+str(index[result[0]]))
    label.pack()
    label1 = Label( root, text="Acessed By : "+val)
    label1.pack()
    
    img = Image.open(x)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()
btn = Button(root, text='open image', command=open_img).pack()
root.mainloop()