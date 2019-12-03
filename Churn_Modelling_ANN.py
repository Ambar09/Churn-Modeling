#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 20:19:13 2019

@author: ambardubey
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_clipboard()
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Creating ANN
import keras
from keras.models import Sequential #to initialize ANN
from keras.layers import Dense #to bulid layers of ANN

# Initialising the ANN
ANN = Sequential()

#Adding the input layer and the 1st hidden layer
#output_dim -> number of nodes in the 1st hidden layer
#init -> uniform random initialization of weights (stochastic gradient descent)
#activation -> activation function
#input_dim -> number of independent variables which is input layer
ANN.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

#Adding the 2nd layer
ANN.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
ANN.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
#adam for applying stochastic gradient descent
#loss parameter to find the cost function and optimize the weights accordingly
#metrics on which our model will improve itself 
ANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
#nb_epoch = number of times we are training the ANN on whole dataset
#batch size = the size after which weights will be reinitialized
ANN.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Predicting the Test set results
y_pred = ANN.predict(X_test)
y_pred = (y_pred > 0.5) #Since y_pred should be binary

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

new_pred = ANN.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred = (new_pred>0.5)