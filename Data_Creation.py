# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:58:43 2023

@author: Anil
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

start = time.process_time()


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
m = 100
input_for_a = np.linspace(5,10,num = m)
input_for_b = np.linspace(5,10, num = m)

aa,bb = np.meshgrid(input_for_a,input_for_b)

a_list = []
b_list = []
opt_list = []
x_val_list = []
y_val_list = []
z_val_list = []

# generate data for parameter a & b and solve the problem in each loop

for i in range(aa.shape[1]):
    a.value = aa[0,i]
    for j in range(bb.shape[1]):
        b.value = bb[j,i]   
        prob.solve(gp=True)
# save the value in lists 
        a_list.append(a.value)
        b_list.append(b.value)
        opt_list.append(prob.value)
        x_val_list.append(x.value)
        y_val_list.append(y.value)
        z_val_list.append(z.value)

# go through the mesh grid of a and b and solve the problem for each combination
# split the data into training and validation data
x = np.array(x_val_list).reshape(-1,1)
y = np.array(y_val_list).reshape(-1,1)
z = np.array(z_val_list).reshape(-1,1)
xyz = np.concatenate((x,y,z),axis = 1)
a = np.array(a_list).reshape(-1,1)
b = np.array(b_list).reshape(-1,1)
ab = np.concatenate((a,b),axis = 1)
#transform the lists into arrays and concatenate the input data(parameters) and the output data(solution of x,y,z) in order to generate training and validation data        
#print(opt_list)
#print(x_val_list)
#print(y_val_list)
#print(z_val_list)


ab_train,ab_valid,xyz_train,xyz_valid = train_test_split(ab,xyz,test_size=0.25,random_state=42)

np.savez('Data',ab_train,ab_valid,xyz_train,xyz_valid)

np.load('Data.npz')