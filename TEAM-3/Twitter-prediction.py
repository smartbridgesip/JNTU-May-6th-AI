from tkinter import *
from keras.models import load_model

import pickle
model = load_model('mymodel.h5')
with open('CountVectorizer','rb') as file:
    cv=pickle.load(file)

top = Tk()
top.geometry("550x300+300+150")
top.resizable(width=True, height=True)

L1 = Label(top, text="Enter Text For Prediction")
L1.pack()
E1 = Entry(top, bd =5)
E1.pack()

def predict1():
   print("Prediction on progress..")
   entered_input=E1.get()
   print("Entered Input",entered_input)
   entered_input=cv.transform([entered_input])
   print(entered_input)
   y_pred=model.predict(entered_input)
   y_pred=(y_pred>0.5)
   print(y_pred)
   if y_pred==True:
       text="The input is a POSITIVE sentiment"
       
   else:
       text="The input is a NEGATIVE sentiment"

   L2 = Label(top, text="Prediction: "+text)
   L2.pack()



B = Button(top, text ="Predict", command = predict1)
B.pack(pady=10)

top.mainloop()

