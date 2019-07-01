#!/usr/bin/env python
# coding: utf-8

# In[8]:


from tkinter import *
root = Tk()
root.configure(background = 'grey')
import numpy as np
from keras.preprocessing import image
from PIL import ImageTk, Image
from tkinter import filedialog
import os
from keras.models import load_model
classifier = load_model(r'C:\Users\Teja\Documents\python example\project\exg.h5')
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

root.geometry("550x400+500+250")
root.resizable(width=True, height=True)

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

def open_img():
    root2 = Tk()
    root2.configure(background = 'grey')
    root2.geometry("550x400+500+250")
    root2.resizable(width=True, height=True)
    x = openfn()
    test_image = image.load_img(x, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict_classes(test_image)
    print(result)
    index= ['Banana','Cactus fruit','Chestnut','Dates','Kohlrabi']
    print(index[result[0]])
    global label 
    label= Label( root2,text="\n\nPrediction : "+str(index[result[0]]),font="Times 28",bg="grey",fg="cyan")
    if result==[0]:
        label2= Label(root2,text="\n Nutrients of "+str(index[result[0]]),font="Times 24",bg="grey", fg="yellow")
        label3= Label(root2,text="\nCabohydrates:23g \nProtien:1.1g \nVitamin A:1 % \nVitamin C:14% \nIron:1%\n\n",font="Times 16",bg="grey")
        label.pack_forget()
    elif result==[1] :
        label2= Label(root2,text="\n Nutrients of "+str(index[result[0]]),font="Times 24",bg="grey", fg="orange")
        label3= Label(root2,text="\nCabohydrates:9.57g \nProtien:0.73g \nVitamin A:1.5 % \nVitamin C:23 % \nCalcium:6 % \nIron:4 %\n\n",font="Times 16",bg="grey")
    elif result==[2] :
        label2= Label(root2,text="\n Nutrients of "+str(index[result[0]]),font="Times 24",bg="grey",fg="red")
        label3= Label(root2,text="\nCabohydrates:28g \nProtien:2g \nVitamin C:44% \nCalcium:4% \nIron:9%\n\n",font="Times 16",bg="grey")
    elif result==[3] :
        label2= Label(root2,text="\n Nutrients of "+str(index[result[0]]),font="Times 24",bg="grey",fg="black")
        label3= Label(root2,text="\nCabohydrates:5.33g \nProtien:0.17g \nVitamin B-6: 0.012 mg \nPotassium: 47 mg \nIron:0.07mg\n\n",font="Times 16",bg="grey")
    else :
        label2= Label(root2,text="\n Nutrients of "+str(index[result[0]]),font="Times 24",bg="grey",fg="green")
        label3= Label(root2,text="\nCabohydrates:23g \nProtien:1.1g \nVitamin A:1 % \nVitamin C:14% \nIron:1% \n\n",font="Times 16",bg="grey")
    label.pack()
    label2.pack()
    label3.pack()
    img = Image.open(x)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    label4=Label(root,text="\n uploaded image is:\n").pack()
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()

    
btn2 = Button(root, text='upload an image', command=open_img,relief=GROOVE ,width=30)
btn2.pack()
    
root.mainloop()


# In[ ]:




