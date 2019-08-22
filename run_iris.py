# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 23:17:12 2018

@author: al2357
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neural_network as nn
import test_nn as t_nn
import custom_plot as cplt

# load and prepare data #1 - minst
#pandas.core.frame.DataFrame
train_set = pd.read_csv("data/mnist_train.csv", header=None, memory_map=True, nrows=2000)
train_samples = np.array(train_set.iloc[:, 1:]).T
train_labels_raw = np.array(train_set.iloc[:, 0])
train_labels_bin = np.zeros((10, len(train_labels_raw)))
train_labels_bin[train_labels_raw, range(0, len(train_labels_raw))] = 1
network_structure = [500, 100, 10]
tct = [0.8, 0.1, 0.1]

# =============================================================================
# # load and prepare data #2 - Iris
# # SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
# train_set = pd.read_csv("data/iris.csv") #, header=None) #, memory_map=True)
# # shuffle, reset index and drop the old index
# train_set = train_set.sample(frac=1).reset_index(drop=True)
# 
# train_set.loc[train_set["Species"]=="Iris-setosa","Species"]=2
# train_set.loc[train_set["Species"]=="Iris-versicolor","Species"]=1
# train_set.loc[train_set["Species"]=="Iris-virginica","Species"]=0
# train_labels_text = np.array(["Virginica", "Versicolor", "Setosa"])
# 
# # labels as integers 0-2
# train_labels_raw = np.array(train_set.iloc[:, 4]).reshape(train_set.shape[0],1).astype('uint8')
#     
# # binary labels
# train_labels_bin = np.zeros((3, len(train_labels_raw)))
# train_labels_bin[train_labels_raw.T, range(0, len(train_labels_raw))] = 1
# 
# # get train samples without labels
# train_samples = np.array(train_set.iloc[:, :4]).T
# 
# network_structure = [10, 5, 3]
# 
# # train, cv, test sets
# tct = [0.6, 0.2, 0.2]
# =============================================================================

nn1 = nn.neural_network(network_structure, \
                        train_samples, \
                        train_labels_bin, \
                        tct, \
                        load_parameters=False)

nn1.learn(gradient_check=True)
local_nn1 = nn1.__dict__


# =============================================================================
# #load and prepare data #3 - binary dataset
# iris = pd.read_csv("iris.csv", header=None, memory_map=True)
# #Create numeric classes for species (0,1,2) 
# iris.loc[iris[4]=="Iris-setosa",4]=2
# iris.loc[iris[4]=="Iris-versicolor",4]=1
# iris.loc[iris[4]=="Iris-virginica",4]=0
# 
# iris=np.array(iris[iris[4]!=2])
# iris_data = iris[:,0:4]
# iris_labels = iris[:,4].reshape((iris.shape[0],1))
# 
# train_samples = iris_data.T
# train_labels_bin = iris_labels.astype('uint8').T
# 
# #plot data
# plt.scatter(train_samples[2, :], train_samples[3, :], c=train_labels_bin[0, :], s=40, cmap=plt.cm.Spectral)
# plt.title("IRIS DATA | Blue - Versicolor, Red - Virginica ")
# plt.xlabel('Petal Length')
# plt.ylabel('Petal Width')
# plt.show()
# network_structure = [6, 1]
# =============================================================================

# =============================================================================
# # instantiate neural network
# tct = [0.6, 0.2, 0.2]
# nn1 = nn.neural_network(network_structure, train_samples, train_labels_bin, tct, load_parameters=False)
# # =============================================================================
# # normalized_samples = nn1.train_samples
# # plt.scatter(normalized_samples[0, :], normalized_samples[1, :], c=train_labels_raw.flatten(), s=40, cmap=plt.cm.Spectral)
# # plt.show()
# # plt.scatter(normalized_samples[2, :], normalized_samples[3, :], c=train_labels_raw.flatten(), s=40, cmap=plt.cm.Spectral)
# # plt.show()
# # =============================================================================
# nn1.learn()
# train_err = nn1.get_train_error();
# print("Training error: "+str(train_err))
# cv_err = nn1.get_cv_error()
# print("CV error: "+str(cv_err))
# test_err = nn1.get_test_error()
# print("Test error: "+str(test_err))
# =============================================================================


# =============================================================================
# # instantiate neural network
# tct = [0.6, 0.2, 0.2]
# nn1 = nn.neural_network(network_structure, train_samples, train_labels_bin, tct, load_parameters=False)
# # =============================================================================
# # normalized_samples = nn1.train_samples
# # plt.scatter(normalized_samples[0, :], normalized_samples[1, :], c=train_labels_raw.flatten(), s=40, cmap=plt.cm.Spectral)
# # plt.show()
# # plt.scatter(normalized_samples[2, :], normalized_samples[3, :], c=train_labels_raw.flatten(), s=40, cmap=plt.cm.Spectral)
# # plt.show()
# # =============================================================================
# nn1.learn()
# train_err = nn1.get_train_error();
# print("Training error: "+str(train_err))
# cv_err = nn1.get_cv_error()
# print("CV error: "+str(cv_err))
# test_err = nn1.get_test_error()
# print("Test error: "+str(test_err))
# =============================================================================
