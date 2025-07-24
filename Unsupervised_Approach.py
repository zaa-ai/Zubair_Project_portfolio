# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:11:46 2023

@author: thomas
"""
import matplotlib as mp
import cvxpy as cp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
import time
import random

# training and validation data were created in "Data_Creation_script" and are saved and then loaded to have reproducible results and constant conditions

Data = np.load('Data.npz')
ab_train = Data['arr_0']
ab_valid = Data['arr_1']
xyz_train = Data['arr_2']
xyz_valid = Data['arr_3']


start = time.process_time()



m = 30


################################################################################

# unsupervised approach - define a penalty function with the constraints so that the problem is unconstrained
# the unsupervised variables must be labeled with "us" in addition

# according to the provided paper the penalty function has the following structure: p(x,y,z,a,b) = 1/(x*y*z)+beta*(x*y+x*z+y*z-a)+beta*(y**b-x)
# according to the paper: "Unsupervised Learning for Parametric Optimization" from Nikbakht beta should be chosen by crossvalidation
# and beta has the same value for each constraint. Values for Beta are (0.01,0.1,1,10,100)
# this is the basic approach for hardwiring the constraints into the structure of the NN itself, but the constraints should be basically negative
# and therefore we will use the relu function so that in order the penalty terms become zero when they are not violated. 
# If the constraints are vioalted, the corresponding penalty term is added to the value of the custom loss function

# custom loss function
def penalty_function (xyz,ab):
    result = 1/(xyz[:,0]*xyz[:,1]*xyz[:,2]) + tf.nn.relu(xyz[:,0]*xyz[:,1] + xyz[:,0]*xyz[:,2] + xyz[:,1]*xyz[:,2] -ab[:,0])**3 + tf.nn.relu(xyz[:,1]**ab[:,1] - xyz[:,0])**3 
    return result

# initialize the model
# for the unsupervised approach there is no regularizer necessary, since there are no labelled data that can be compared with it 
model = keras.Sequential(name = 'unsupervised_NN')
model.add(keras.layers.Input(shape = (ab_train.shape[1],) ,name = 'input'))
model.add(keras.layers.Dense(units = 50, activation = 'relu',name = 'input-layer',activity_regularizer=tf.keras.regularizers.L1(0.001))) 
model.add(keras.layers.Dense(units = 50, activation = 'relu',name = 'hidden-layer1',activity_regularizer=tf.keras.regularizers.L1(0.001)))
model.add(keras.layers.Dense(units = 50, activation = 'relu',name = 'hidden-layer2',activity_regularizer=tf.keras.regularizers.L1(0.001))) #slightly better results without regularization, but according to the Paper there is a risk of overfitting
model.add(keras.layers.Dense(units = 50, activation = 'relu',name = 'hidden-layer3',activity_regularizer=tf.keras.regularizers.L1(0.001)))
model.add(keras.layers.Dense(units = 3, activation = tf.exp, name = 'output',activity_regularizer=tf.keras.regularizers.L1(0.001))) #exponential layer, so that x,y,z remain positive 

model.summary()


print("{} s taken for startup".format(time.process_time() - start)) #implementation of time measurement 
# in the following a custom training loop according to figure 1 in the task sheet is implemented

train_loss_results = [] #a list for the losses during the training epochs

num_epochs = 50

# definition of loss function including a forward pass
def loss(model,ab):
    xyz_pred = model(ab)
    return penalty_function(xyz_pred,ab)

# gradients of the loss value dependent on the weights and biases are used to optimize the model 
def grad(model,ab,xyz):
    with tf.GradientTape() as tape:
        loss_value = loss(model,ab)
    return loss_value, tape.gradient(loss_value,model.trainable_variables)

@tf.function
# implementation of one single training step
# first gradient and loss value are calculated and then they are passed to the optimizer
# optimizer uses the computed gradients for upgrading the weights and biases so that our custom loss function is minimized
def train_step(ab, xyz_pred_us):
    loss_value,grads = grad(model,ab_train,xyz_pred_us)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value

def chunks(lst, n):
    chunked_lst = []
    for i in range(0, len(lst), n):
        chunked_lst.append(lst[i:i + n])
    return chunked_lst   

# the training data set (ab_train) is chunked in a list of (n = ) 50 elements each containing 150 elements, so that the batch size is 150
ab_train_chunks  = chunks(ab_train,50)   
before_train = time.process_time()
opt = tf.keras.optimizers.Adam(learning_rate=0.0001) 

# mini-batch stochastic gradient descent
for epoch in range(num_epochs):   # loop for the epochs
    epoch_start = time.process_time()     
    epoch_loss_avg = tf.keras.metrics.Mean() # metrics to define the mean loss for the epochs     
    rand_chunk = random.randint(0,len(ab_train_chunks)-1) # creation of a ranodm integer, so that a random batch is chosen
    for ab in ab_train_chunks[rand_chunk]: # a randomly chosen batch is used for a training step
        ab = ab.reshape(1,2)
        xyz_pred_us = model(ab) #forward pass
        loss_value = train_step(ab,xyz_pred_us)
        epoch_loss_avg.update_state(loss_value) # updating the loss after the train step
        

    train_loss_results.append(epoch_loss_avg.result())    
    print("Epoch {}: Loss: {}".format(epoch+1,epoch_loss_avg.result()))
   

train_time = time.process_time() - before_train
print("{:.2f}s for training".format(train_time))
print("average of {:.2f}s per epoch".format(train_time/num_epochs) )

# error value to evaluate the performance of the model and also as an orientation point for hyperparameter tuning

xyz_pred_us2 = model(ab_valid) 
mse = tf.keras.losses.MeanSquaredError()
error = mse(xyz_valid,xyz_pred_us2)

# Visualization of training efficiency
# in the end we create a plot containing the development of the loss function and the different predicted values against the validated data

fig = plt.figure()
fig.set_figheight(50)
fig.set_figwidth(100)

ax1 = plt.subplot2grid(shape=(4,2),loc=(0,0),colspan=1)
ax1.semilogy(label='Training efficiency_unsupervised')
ax1.legend()
ax1.plot(train_loss_results)
ax1.set_xlabel('epochs [-]')
ax1.set_ylabel('loss [-]')

# Visualization of performance -> the validation data for x,y,z from the solver are plotted against the predicted values of x,y,z

ax2 = plt.subplot2grid(shape=(4,2),loc=(1,0),colspan=1)
ax2 = sb.regplot(x = xyz_valid[:,0],y = xyz_pred_us2[:,0])
ax2.set(xlabel = 'x_valid [-]', ylabel = 'x_pred_us2 [-]', title = 'x_pred over x_valid (unsupervised)')

ax3 = plt.subplot2grid(shape=(4,2),loc=(2,0),colspan=1)
ax3 = sb.regplot(x = xyz_valid[:,1],y = xyz_pred_us2[:,1])
ax3.set(xlabel = 'y_valid [-]', ylabel = 'y_pred [-]', title = 'y_pred over y_valid (unsupervised)')

ax4 = plt.subplot2grid(shape=(4,2),loc=(3,0),colspan=1)
ax4 = sb.regplot(x = xyz_valid[:,2],y = xyz_pred_us2[:,2])
ax4.set(xlabel = 'z_valid [-]', ylabel = 'z_pred [-]', title = 'z_pred over z_valid (unsupervised)')

plt.savefig('unsupervised-elu')
model.save('C:/Users/thomas/Desktop/UNI/Master-CIW/2.Semester/Machine Learning Methods for Engineers/Übungen/Semesterprojekt-Unsupervised parametric optimization/unsupervised')
