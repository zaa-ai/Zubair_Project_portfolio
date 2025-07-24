# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 21:01:10 2023

@author: Zoe
"""
"""
Spyder Editor

This is a temporary script file.
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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


# Variables
x = cp.Variable(pos = True)
y = cp.Variable(pos = True)
z = cp.Variable(pos = True)

# Parameter 
a = cp.Parameter(pos = True)
b = cp.Parameter()

# Constraints
constraints = [x*y + x*z + y*z  <= a,
               cp.power(y,b)  <= x]

# Objective function
obj_fun = 1/(x*y*z)

# Objective
obj = cp.Minimize(obj_fun)

prob = cp.Problem(obj, constraints)

# Parameter 
m = 30
input_for_a = np.linspace(1,30,num = m)
input_for_b = np.linspace(-15,15, num = m)

opt_list = []
x_val_list = []
y_val_list = []
z_val_list = []

# generate data for parameter a & b and solve the problem in each loop

for i in range(len(input_for_a)): 
    a.value = input_for_a[i]  
    b.value = input_for_b[i]  
    prob.solve(gp=True)
    #save the value in lists
    opt_list.append(prob.value)
    x_val_list.append(x.value)
    y_val_list.append(y.value)
    z_val_list.append(z.value)
# go through the parameter vectors of a and b and solve the problem for each combination
        
print(opt_list)
print(x_val_list)
print(y_val_list)
print(z_val_list)

# split the data into training and validation data
# transform the lists into arrays and concatenate the input data(parameters) and the output data(solution of x,y,z) in order to generate training and validation data
a = np.array(input_for_a).reshape(-1,1)
b = np.array(input_for_b).reshape(-1,1)
ab = np.concatenate((a,b),axis=1) 

x = np.array(x_val_list).reshape(-1,1)
y = np.array(y_val_list).reshape(-1,1)
z = np.array(z_val_list).reshape(-1,1)
xyz = np.concatenate((x,y,z),axis=1)

ab_train,ab_valid,xyz_train,xyz_valid = train_test_split(ab,xyz,test_size=0.25,random_state=42)

###############################################################################
##supervised approach - train a model with supervised training and evaluate its performance 


# initialize the model first 
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.svm import SVC  #need to define diffrent hyperparamerts for SVC as an estimator
#hyperparamters to tune:
   # 1) units: Refers to the number of neurons or nodes in a particular layer of the network. Each unit receives inputs from the previous layer, performs a computation, and passes its output to the next layer.
   # 2) num_layers: This hyperparameter controls the depth and complexity of the model.
   # 3) learning_rate: Determines the step size at which the optimization algorithm adjusts the weights of a neural network during training. It controls how quickly or slowly the model learns from the data.
# regarding the paper the neurons in the input-layer should be m^2 and in the hidden layer between m and m^2 (m = input dimension)
def model(num_layers=1, units=16, learning_rate=0.001):
    model = keras.Sequential(name = 'supervised_NN')
    for i in range(num_layers):  #a loop is used to add the specified number of hidden layers, each with the specified number of units
        model.add(keras.layers.Input(shape = (ab_train.shape[1],) ,name = 'input-layer'))
        model.add(keras.layers.Dense(units = m**2, activation = 'relu',name = 'hidden-layer1',activity_regularizer=tf.keras.regularizers.L2(0.001)))
        model.add(keras.layers.Dense(units = m**2, activation = 'relu',name = 'hidden-layer2',activity_regularizer=tf.keras.regularizers.L2(0.001)))
        model.add(keras.layers.Dense(units = m**2, activation = 'relu',name = 'hidden-layer3',activity_regularizer=tf.keras.regularizers.L2(0.001)))
        model.add(keras.layers.Dense(units = 3, name = 'output',activity_regularizer=tf.keras.regularizers.L2(0.001)))
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),loss = tf.keras.losses.MSE, metrics = ['accuracy'])
    return model

# Wrap the Keras model for compatibility with scikit-learn GridSearchCV
regressor = KerasRegressor(build_fn=model, verbose =2)
# Since Keras models have their own Interface and are not directly compatible with scikit-learn, 
#we need to use the KerasRegressor wrapper to bridge the gap.
# the verbose parameter is used to control the amount of information displayed during the training process.
#The verbose parameter accepts different integer values that determine the verbosity level.

# Define the hyperparameters to tune
param_grid = {
    'num_layers': [1, 2, 3, 4],
    'units': [16, 32, 64, 128, 256],
    'learning_rate': [0.001, 0.01, 0.1, 0.00001]
}

# Perform grid search using cross-validation accorss the grid to get the best combination
#Grid search systematically trains and evaluates the model for all possible combinations of the specified hyperparameter values.
grid = GridSearchCV(estimator=regressor, param_grid=param_grid,cv=10) #cv, (cross validation) dividing data into 3 or 5 subsets
grid_result = grid.fit(ab_train, xyz_train)  

#with cv =3, we get best parameters as{'learning_rate': 0.1, 'num_layers': 1, 'units': 64} Best Accuracy:  -3.8745344479878745
#with cv =5, we get best parameters as{'learning_rate': 0.001, 'num_layers': 1, 'units': 128}Best Accuracy:  -3.6533137798309325
##with cv =10, we get best parameters as{'learning_rate': 0.001, 'num_layers': 1, 'units': 32}Best Accuracy:  -4.028146785497666


# Print the best parameters and score
print("Best Parameters: ", grid_result.best_params_)
print("Best Accuracy: ", grid_result.best_score_) 
#print("Best Accuracy:",grid_result.score(ab_test,xyz_test)
#The best score indicates the highest value achieved by the scoring metric among all the
#different combinations of hyperparameters that were tested. 

# Extract the hyperparameters and scores from the grid search results
params = grid_result.cv_results_['params']
mean_scores = grid_result.cv_results_['mean_test_score']

# Extract the values of each hyperparameter
units_values = [param['units'] for param in params]
learning_rate_values = [param['learning_rate'] for param in params]

# Plot the hyperparameters against the scores
#Plotting hyperparameters against the score provides a visual representation of how
# different values of hyperparameters impact the performance of the mode
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(units_values, mean_scores, marker='o')
plt.xlabel('Units')
plt.ylabel('Mean Test Score')
plt.title('Hyperparameter Tuning: Units')

plt.subplot(1, 2, 2)
plt.plot(learning_rate_values, mean_scores, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Mean Test Score')
plt.title('Hyperparameter Tuning: Learning Rate')

plt.tight_layout()
plt.show()