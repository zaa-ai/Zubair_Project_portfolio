# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

author: thomas
"""

#hints: use training data with parameters from 5-20, do not sample from non-uniform distributions for a, values for a close to 0 changes solution a lot
#a&b are trainig data for NN in both cases (supervised: L = MSE(Mean Squared Error)) (unsupervised: solve a penalty function )


##supervised approach - generation of training and validation data

import matplotlib as mp
import cvxpy as cp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb

# training and validation data were created in "Data_Creation_script" and are saved and then loaded to have reproducible results and constant conditions

Data = np.load('Data.npz')
ab_train = Data['arr_0']
ab_valid = Data['arr_1']
xyz_train = Data['arr_2']
xyz_valid = Data['arr_3']

m = 30


###############################################################################
##supervised approach - train a model with supervised training and evaluate its performance 


# initialize the model first 

# the neurons will be chosen similar to the ones that are chosen in the paper
model = keras.Sequential(name = 'supervised_NN')
model.add(keras.layers.Input(shape = (ab_train.shape[1],) ,name = 'input'))
model.add(keras.layers.Dense(units = m**2, activation = 'relu',name = 'input-layer',activity_regularizer=tf.keras.regularizers.L2(0.001)))
model.add(keras.layers.Dense(units = m**2, activation = 'relu',name = 'hidden-layer1',activity_regularizer=tf.keras.regularizers.L2(0.001)))
model.add(keras.layers.Dense(units = m**2, activation = 'relu',name = 'hidden-layer2',activity_regularizer=tf.keras.regularizers.L2(0.001)))
model.add(keras.layers.Dense(units = 3, name = 'output',activity_regularizer=tf.keras.regularizers.L2(0.001)))
#regularizer only in order to prevent overfitting

# compile the model with adam optimizer and MSE-loss function
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),loss = tf.keras.losses.MSE, metrics = [keras.metrics.MeanSquaredError()])

# train the model 
history = model.fit(ab_train,xyz_train,batch_size = 50,epochs = 50,validation_data = (ab_valid,xyz_valid))

# predict some values with ab_validation data
xyz_pred = model.predict(ab_valid)
# error value to evaluate the performance of the model and also as an orientation point for hyperparameter tuning
mse = tf.keras.losses.MeanSquaredError()
error_supervised = mse(xyz_valid,xyz_pred)
# Visualization of training efficiency
# in the end we create a grid for the subplots of the evaluation

fig = plt.figure()
fig.set_figheight(50)
fig.set_figwidth(100)

ax1 = plt.subplot2grid(shape=(4,2),loc=(0,0),colspan=1)
ax1.semilogy(history.history['loss'], label='Training efficiency_supervised')
ax1.legend()
ax1.set_xlabel('epochs [-]')
ax1.set_ylabel('loss [-]')

# Visualization of performance -> the validation data for x,y,z from the solver are plotted against the predicted values of x,y,z

ax2 = plt.subplot2grid(shape=(4,2),loc=(1,0),colspan=1)
ax2 = sb.regplot(x = xyz_valid[:,0],y = xyz_pred[:,0])
ax2.set(xlabel = 'x_valid [-]', ylabel = 'x_pred [-]', title = 'x_pred over x_valid (supervised)')

ax3 = plt.subplot2grid(shape=(4,2),loc=(2,0),colspan=1)
ax3 = sb.regplot(x = xyz_valid[:,1],y = xyz_pred[:,1])
ax3.set(xlabel = 'y_valid [-]', ylabel = 'y_pred [-]', title = 'y_pred over y_valid (supervised)')

ax4 = plt.subplot2grid(shape=(4,2),loc=(3,0),colspan=1)
ax4 = sb.regplot(x = xyz_valid[:,2],y = xyz_pred[:,2])
ax4.set(xlabel = 'z_pred [-]', ylabel = 'z_val [-]', title = 'z_pred over z_valid (supervised)')

plt.savefig('supervised')
model.save('C:/Users/thomas/Desktop/UNI/Master-CIW/2.Semester/Machine Learning Methods for Engineers/Übungen/Semesterprojekt-Unsupervised parametric optimization/unsupervised')

# error value to evaluate the performance of the model and also as an orientation point for hyperparameter tuning










