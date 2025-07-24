# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 11:08:31 2023

@author: Zubair
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
from Data_Creation2 import ab_train,ab_valid,xyz_train,xyz_valid
import random

start = time.process_time()

# Parameter 
m = 30

# initialize the model
# for the unsupervised approach there is no regularizer necessary, since there are no labelled data that can be compared with it 
model = keras.Sequential(name = 'unsupervised_NN')
model.add(keras.layers.Input(shape = (ab_train.shape[1],) ,name = 'input-layer'))
model.add(keras.layers.Dense(units = m**2, activation = 'relu',name = 'hidden-layer1',activity_regularizer=tf.keras.regularizers.L1(0.001))) 
model.add(keras.layers.Dense(units = m**2, activation = 'relu',name = 'hidden-layer2',activity_regularizer=tf.keras.regularizers.L1(0.001)))
model.add(keras.layers.Dense(units = m**2, activation = 'relu',name = 'hidden-layer3',activity_regularizer=tf.keras.regularizers.L1(0.001))) #for Ali L1 worked better
model.add(keras.layers.Dense(units = m**2, name = 'hidden-layer4',activity_regularizer=tf.keras.regularizers.L1(0.001)))
model.add(keras.layers.Dense(units = 3, activation = tf.exp, name = 'output',activity_regularizer=tf.keras.regularizers.L1(0.001))) #first linear and then exponential layer

model.summary()

# Assuming you have trained the model and obtained the predictions
xyz_pred_us = model.predict(ab_train)  # Predicted values of x, y, z

# Convert 'xyz_pred_us' and 'ab_train' to TensorFlow tensors if needed
xyz_pred_us = tf.convert_to_tensor(xyz_pred_us)
ab_train = tf.convert_to_tensor(ab_train)

xyz_pred_us = tf.cast(xyz_pred_us, dtype=tf.float32)  # Convert 'xyz_pred_us' to float32
ab_train = tf.cast(ab_train, dtype=tf.float32)        # Convert 'ab_train' to float32


def loss_function(xyz, ab):
    # Inequality constraints: (xy + xz + yz) − a ≤ 0 and yb − x ≤ 0
    slack1 = tf.nn.relu((xyz[:, 0] * xyz[:, 1] + xyz[:, 0] * xyz[:, 2] + xyz[:, 1] * xyz[:, 2]) - ab[:, 0])  # Slack variable for (xy + xz + yz) − a ≤ 0
    slack2 = tf.nn.relu(xyz[:, 1] * ab[:, 1] - xyz[:, 0])  # Slack variable for yb − x ≤ 0
    
    # Reformulated equality constraints using slack variables:
    # (xy + xz + yz) − slack1 = a and yb − slack2 = x
    equality_loss = tf.reduce_mean(tf.square((xyz[:, 0] * xyz[:, 1] + xyz[:, 0] * xyz[:, 2] + xyz[:, 1] * xyz[:, 2]) - slack1 - ab[:, 0])) + tf.reduce_mean(tf.square(xyz[:, 1] * ab[:, 1] - slack2 - xyz[:, 0]))
    
    # Penalty function for other constraints
    penalty_loss = 1 / (xyz[:, 0] * xyz[:, 1] * xyz[:, 2]) + 10 * tf.nn.relu(xyz[:, 0] * xyz[:, 1] + xyz[:, 0] * xyz[:, 2] + xyz[:, 1] * xyz[:, 2] - ab[:, 0])**3 + 10 * tf.nn.relu(xyz[:, 1]**ab[:, 1] - xyz[:, 0])**3

    # Combine the losses with appropriate weights
    total_loss = equality_loss + penalty_loss

    return total_loss


# Calculate the loss using the updated loss function
loss_value = loss_function(xyz_pred_us, ab_train)

print("{} s taken for startup".format(time.process_time() - start))
# compile the model with adam optimizer and use the penalty function from before to defin

train_loss_results = []

num_epochs = 10

def grad(model,ab,xyz):
    with tf.GradientTape() as tape:
        loss_value = loss_function(model,ab)
    return loss_value, tape.gradient(loss_value,model.trainable_variables)

@tf.function
def train_step(ab, xyz_pred_us):
    loss_value,grads = grad(model,ab_train,xyz_pred_us)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value

def chunks(lst, n):
    chunked_lst = []
    for i in range(0, len(lst), n):
        chunked_lst.append(lst[i:i + n])
    return chunked_lst   

ab_train_chunks  = chunks(ab_train,50)   
before_train = time.process_time()
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)


batch_size = 32  # Choose your desired batch size
#Within the training loop, instead of iterating over individual data points, 
#we iterate over the batches of the dataset using for ab_batch in ab_train_batches. 
#The forward pass, loss computation, gradient calculation, 
#and parameter updates are now performed on batches instead of individual data points.
# ...
ab_train = tf.cast(ab_train, tf.float32)
ab_valid = tf.cast(ab_valid, tf.float32)

ab_train_batches = tf.data.Dataset.from_tensor_slices(ab_train).batch(batch_size)

for epoch in range(num_epochs):
    epoch_start = time.process_time()
    epoch_loss_avg = tf.keras.metrics.Mean()
    
    for ab_batch in ab_train_batches:
        with tf.GradientTape() as tape:
            xyz_pred_us_batch = model(ab_batch)
            loss_value = loss_function(xyz_pred_us_batch, ab_batch)
        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        
        epoch_loss_avg.update_state(loss_value)

    train_loss_results.append(epoch_loss_avg.result())
    print("Epoch {}: Loss: {}".format(epoch+1, epoch_loss_avg.result()))
    #print("{} s spent in epoch".format(time.process_time() - epoch_start))

train_time = time.process_time() - before_train
print("{:.2f}s for training".format(train_time))
print("average of {:.2f}s per epoch".format(train_time/num_epochs) )

xyz_pred_us2 = model(ab_valid)
mse = tf.keras.losses.MeanSquaredError()
error = mse(xyz_valid,xyz_pred_us2)