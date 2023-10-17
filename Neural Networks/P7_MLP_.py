# -*- coding: utf-8 -*-
"""

"""
# Old versions
#conda install -c conda-forge keras
#conda install -c conda-forge/label/broken keras
#conda install -c conda-forge/label/cf201901 keras
#conda install -c conda-forge tensorflow=1.14

# Lastest version
#pip install keras
#pip install tensorflow

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate dummy data
import numpy as np
from sklearn.metrics import r2_score


#%% UNIVARIATE EXAMPLE: Generation of data in 1D
m = 1000;
X = np.linspace(-1,1,num=m)
X = np.reshape(X,(m,1))

Y = X**2 # Y = X^21
# Y = 2*X**3+2*X**2 # Y = 2X^3+2X^2
# Y = 4*X*np.sin(10*X)+X*np.exp(X)*np.cos(20*X) # Y = 4X*sin(10X)+X*exp^(X)*cos(20X)

[m,ne] = np.shape(X)

#from sklearn import preprocessing
#X = preprocessing.scale(X)
#Y = preprocessing.scale(Y)

fig = plt.figure()
plt.plot(X,Y, linewidth=3)
plt.grid()

#%% Creation of the neural network for the 1D problem

# https://keras.io/getting-started/sequential-model-guide/
model = Sequential(
    [tf.keras.Input(shape = (1,),name='input_layer'),
    Dense(10, activation='tanh',name='hidden_layer1'),
    Dense(1, activation='linear',name='output_layer')
    ], name = 'my_model'
    )

# https://keras.io/activations/

# https://keras.io/optimizers/
opt_object = SGD(lr=0.01, momentum=0.5)
# opt_object = SGD(lr=0.001, beta_1=0.9, beta_2=0.999)

#https://keras.io/api/losses/
model.compile(loss='mean_squared_error',
              optimizer=opt_object)

history = model.fit(X, Y, epochs=100)

score = model.evaluate(X, Y)

Yhat = model.predict(X)


J = history.history['loss']
R2_score = r2_score(Y,Yhat)


#%% Visualize the evolution of learning
fig = plt.figure(figsize=(15,8))
plt.plot(np.arange(len(J)),J, linewidth=3)
plt.title('J(W) vs epoch: J(W)=%0.4f'%J[-1])
plt.xlabel('epoch')
plt.ylabel('J(W)')
plt.grid()
plt.show()


# Display of the estimate by the regression
fig = plt.figure(figsize=(15,8))
plt.plot(X[:,0],Y,label='Y real', linewidth=3)
plt.plot(X[:,0],Yhat,label='Y estimated', linewidth=3)
plt.title('Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

fig = plt.figure(figsize=(15,8))
plt.scatter(Y,Yhat)
plt.plot(Y,Y,'k--',linewidth=3)
plt.axis('square')
plt.title('Prediction: R^2=%0.4f'%R2_score)
plt.xlabel('Y')
plt.ylabel('Yhat')
plt.grid()
plt.show()


#%% MULTIVARIABLE EXAMPLE: Generation of 3D data
m = 100;
X = np.linspace(-1,1,num=m)
[x1,x2] = np.meshgrid(X,X)
X = np.c_[x1.ravel(),x2.ravel()]
Y = X[:,0]**2+X[:,1]**2 # Y = X1^2+X2^2
# Y = 2*X[:,0]**3+2*X[:,0]**2+2*X[:,1]**3+2*X[:,1]**2 # Y = 2*X1^3+2*X1^2+2*X2^3+2*X2^2

# X = np.linspace(0,1,num=m)
# [x1,x2] = np.meshgrid(X,X)
# X = np.c_[x1.ravel(),x2.ravel()]
# Y = 4*X[:,0]*np.sin(10*X[:,0])*np.cos(10*X[:,1])+X[:,0]*X[:,1]*np.exp(X[:,0]*X[:,1])*np.cos(20*X[:,0]*X[:,1])


[m,ne] = np.shape(X)
Y = np.reshape(Y,(m,1))

from sklearn import preprocessing
X = preprocessing.scale(X)
Y = preprocessing.scale(Y)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)


#%% Creation of the neural network for the 3D problem

# https://keras.io/getting-started/sequential-model-guide/
model = Sequential(
    [tf.keras.Input(shape = (2,),name='input_layer'),
     Dense(5, activation='tanh', input_dim=2,name='hidden_layer'),
     Dense(1, activation='linear',name='output_layer')
     ]
    )

# https://keras.io/activations/
# model.add(Dense(5, activation='tanh', input_dim=2,name='input_layer'))
# model.add(Dense(1, activation='linear',name='output_layer'))

# https://keras.io/optimizers/
sgd = SGD(lr=0.01, decay=1e-15, momentum=0.0)
model.compile(loss='mean_squared_error',
              optimizer=sgd)

history = model.fit(X, Y,epochs=100)

score = model.evaluate(X, Y)

Yhat = model.predict(X)


J = history.history['loss']
R2_score = r2_score(Y,Yhat)

#%% Visualize the evolution of learning
fig = plt.figure(figsize=(15,8))
plt.plot(np.arange(len(J)),J, linewidth=3)
plt.title('J(W) vs epoch: J(W)=%0.4f'%J[-1])
plt.xlabel('epoch')
plt.ylabel('J(W)')
plt.grid()
plt.show()


# Display of the estimate by the regression
fig = plt.figure(figsize=(15,8))
plt.scatter(Y,Yhat)
plt.plot(Y,Y,'k--',linewidth=3)
plt.axis('square')
plt.title('Prediction: R^2=%0.4f'%R2_score)
plt.xlabel('Y')
plt.ylabel('Yhat')
plt.grid()
plt.show()


fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
ax.scatter(X[:,0], X[:,1], Yhat,c='r')
plt.title('Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()