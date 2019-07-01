from keras.models import load_model
import numpy as np
import pickle
model = load_model('mymodel.h5')
with open('CountVectorizer','rb') as file:
    cv=pickle.load(file)

type(cv)
x_intent="I am a GOOD girl"
x_intent=cv.transform([x_intent])
y_pred=model.predict(x_intent)
y_pred=(y_pred>0.5)
print(y_pred)

