# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:40:12 2019

@author: Yasir hussain
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score




dataset = pd.read_csv('path-to-csv-dataset')
num_classes = 2

#slicing data
X = dataset.iloc[:, 0:10] #index of columns that contains features in dataset 
Y = dataset.iloc[:, 10:11]#index of last column as label 



#scaling dataset
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)

#slicing data for train and test
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)



# one hot encoding
y_train = to_categorical(y_train, num_classes)
Y = pd.DataFrame(y_train)



#model
model = Sequential()
model.add(Dense(30, input_dim=x_train.shape[1],activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# Train the model, iterating on the data in batches of 256 samples 
model.fit(x_train, y_train, epochs = 30, batch_size = 256)

predictions = model.predict(x_test)
predictions = np.round(predictions)
pred = predictions.argmax(1)
pred = pd.DataFrame(pred)



#Accuracy
print ("Accuracy:  ", accuracy_score(y_test,pred))