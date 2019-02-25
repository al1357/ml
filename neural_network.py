# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 23:56:35 2018

@author: al2357
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import float64, int
import math
"""

"""
class neural_network:
    
    data_samples = {}
    labels = {}
    
    # number of rows/training examples
    m = {}
    # number of features
    n = 1
    network_map = None
    # z cache
    cache_z = []
    cache_a = []
    # error cache iteration-error
    cache_train_error = []
    cache_test_error = []
    cache_iterations = []
    # learing rate
    alpha = None
    weights = {}
    bias = {}
    buffer = {}
    
    def __init__(self, network_map=[], data_samples=[], labels=[], tct_split = [1, 0, 0], load_parameters=False):
        """
            training/test sets dimensions are n x m
        """
        np.random.seed(3)
        self.network_map = network_map
        
        all_data_m = data_samples.shape[1]
        self.m['train'] = self.round(all_data_m * tct_split[0])
        self.m['cv'] = self.round(all_data_m * tct_split[1])
        self.m['test'] = self.round(all_data_m * tct_split[2])
        self.n = data_samples.shape[0]
        
        self.data_samples['train'] = data_samples[:, 0:self.m['train']]
        self.labels['train'] = labels[:, 0:self.m['train']]
    
        if self.m['cv'] != 0:
            self.data_samples['cv'] = data_samples[:, self.m['train']:(self.m['train'] + self.m['cv'])]
            self.labels['cv'] = labels[:, self.m['train']:(self.m['train'] + self.m['cv'])]
            
        if self.m['test'] != 0:
            self.data_samples['test'] = data_samples[:, (self.m['train'] + self.m['cv']):]
            self.labels['test'] = labels[:, (self.m['train'] + self.m['cv']):]
        #end if

        # load saved parameters
        if load_parameters:
            self.load_parameters()
        else:   
            self.initialize_parameters('he')
        #end if
        
        self.alpha = 0.05
        
        self.normalize()
    #end
    
    def save_parameters(self):
        for l in self.weights:
            with open('weights_l'+str(l)+'.txt', 'w') as outfile:
                outfile.write('# Weights for layer {0}; shape {1}\n'.format(l, self.weights[l].shape))
                np.savetxt(outfile, self.weights[l])
            #end with
        #end for
        
        with open('bias.txt', 'w') as outfile:
            outfile.write('# Bias dict. length: {0}\n'.format(len(self.bias))) 
            for i in self.bias:
                outfile.write('# Bias for layer {0}; shape {1}\n'.format(i, self.bias[i].shape))
                np.savetxt(outfile, self.bias[i])
            #end for
        #end with
    #end
    
    def load_parameters(self):
        try:
            i = 1
            for nl in self.network_map:
                self.weights[i] = np.loadtxt('weights_l'+str(i)+'.txt')
                i = i + 1
            #end for
            bias = np.loadtxt('bias.txt')
        except Exception as e:
            print("Weights/bias file read error. Initializing random parameters.")
            print("Error msg: "+str(e))
            if not bool(self.weights) or not bool(self.bias):    
                self.initialize_parameters('he')
        else:
            prev_layer_size = 0
            i=1
            for nl in self.network_map:
                self.bias[i] = np.array(bias[prev_layer_size:prev_layer_size + nl]).reshape(nl, 1)
                prev_layer_size += nl
                i += 1
            #end for
        #end else
    #end
        
    def initialize_parameters(self, how='random'):
        previous_layer = self.n
        i = 1
        for l in self.network_map:
            if how == 'he':
                mpr = np.sqrt(2/previous_layer)
            elif how == 'random':
                mpr = 10
            else:
                mpr = 1
            #endif
            current_weights = np.random.randn(l, previous_layer)*mpr
            previous_layer = l
            self.weights[i] = current_weights
            self.bias[i] = np.zeros((l, 1))
            i += 1
    #end
    
    def normalize(self):
        # calculate mean and variance only for training set; use them to normalize the rest
        mean = np.sum(self.data_samples['train'], axis=1, keepdims=True) / self.m['train']
        variance = np.sum(self.data_samples['train']**2, axis=1, keepdims=True) / self.m['train']
        
        for ix in self.data_samples:
            self.data_samples[ix] = (self.data_samples[ix] - mean) / variance
            np.nan_to_num(self.data_samples[ix], False)
    #end
    
    def sigmoid(self, z):
        """Sigmoid function
        prediciton: sigmoid >= 0.5 result 1; sigmoid < 0.5 result 0;
        """
        return 1 / (1 + np.exp(-z))
    #end
    
    def d_sigmoid(self, z):
        """ derivative of the sigmoid function """
        zPrim = self.sigmoid(z)
        return zPrim*(1-zPrim)
    #end
    
    def tanh(self, z):
        """Tanh activation function
        prediction: tanh >= 0 result 1; tanh < 0 result 0;
        """
        return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
    #end
    
    def d_tanh(self, z):
        return 1 - np.power(self.tanh(z), 2)
    #end
    
    def loss(self, kind='train', pred = None):
        if kind == 'train' or kind == 'test' or kind == 'cv':
            labels = self.labels[kind]
        else:
            return 0
        #end if
        
        if kind == 'train':
            pred = self.cache_a[-1]
        elif pred is None:
            return
        #end if
        
        loss = -(np.sum(labels*(np.log(pred)) + (1-labels)*(np.log(1-pred))) / self.m[kind])
        return loss
        
    def forward_propagate(self, kind='train', weights=False, bias=False):
        if kind == 'train' or kind == 'test' or kind == 'cv':
            X = self.data_samples[kind]
        else:
            return 0
        #end if
        if weights == False:
            weights = self.weights
        #end if
        if bias == False:
            bias = self.bias
        #end if
        
        # layersA[0] will be x; layersZ[0] is empty
        self.cache_a = []
        self.cache_z = []
        self.cache_z.append(np.ndarray([]))
        self.cache_a.append(X)
        # Iterate throught layers; e.g. range(2) = [0, 1]; 
        for i in range(1, len(self.network_map)+1):
            # cache_z.shape = (n-next + 1, 1)
            # b's shape is (i+1,1) and it's added to the new matrix before activation fn is applied
            # it is broadcasted to all results along m axis
            layer_z = np.dot(weights[i], self.cache_a[i-1]) + bias[i]
            # Save layerZ in cache for back prop
            self.cache_z.append(layer_z)
            if i == len(self.network_map):
                # Last is sigmoid
                layer_a = self.sigmoid(layer_z)
            else:
                # Non-last are tanh
                layer_a = self.tanh(layer_z)
            np.nan_to_num(layer_a, False)
            self.cache_a.append(layer_a)
        #end for
    #end forwardPropagate
    
    def back_propagate(self, kind='train'):
        if kind != 'train' and kind != 'test' and kind != 'cv':
            return 0
        #end if
        # e.g. range(2, 0) = [2, 1]
        n_depth = len(self.network_map)
        prev_dz = []
        for i in range(n_depth, 0, -1):
            if i == n_depth:
                #print(self.cache_a[i].shape)
                #print(self.train_labels.shape)
                dz = (self.cache_a[i] - self.labels[kind])
            else:
                # i = 1; s1 contain n1 neurons; (n1, m) = (n1, n2) x (n2, m) * (n1, m)
                dz = np.dot(self.weights[i+1].T, prev_dz)*self.d_tanh(self.cache_z[i])
            prev_dz = dz
            dw_temp = np.dot(dz, self.cache_a[i-1].T) / self.m[kind]
            db_temp = np.sum(dz, axis=1, keepdims=True) / self.m[kind]
            self.weights[i] = self.weights[i] - self.alpha*dw_temp
            self.bias[i] = self.bias[i] - self.alpha*db_temp
        #end for
    #end backPropagate
    
    def learn(self, gradient_check = False):
        for i in range(0,9000):
            ##print('----------------------------------------------')
            #print(i)
            self.forward_propagate()
            # optional gradient check
            if gradient_check == True:
                train_error = self.get_train_error()
                approx_error = self.get_approx_train_error()
                error_diff = train_error-approx_error
            self.back_propagate()
            if i%200 == 0:
                self.cache_train_error.append(self.get_train_error())
                ##self.cache_cv_error.append(self.loss())
                self.cache_test_error.append(self.get_test_error())
                self.cache_iterations.append(i)
        #end for
        plt.plot(self.cache_iterations, self.cache_train_error)
        plt.plot(self.cache_iterations, self.cache_test_error)
        plt.legend(['train error', 'test error'], loc='upper left')
        plt.title("Error vs iterations")
        plt.xlabel("iterations")
        plt.ylabel("error")
        plt.show()
        print("Last train error: ",str(self.cache_train_error[-1]))
        print("Last test error: ",str(self.cache_test_error[-1]))
        self.save_parameters()
    #end learn
    
    def get_train_error(self):
        pred = self.predict('train')
        return self.loss('train', pred)
    
    def get_test_error(self):
        pred = self.predict('test')
        return self.loss('test', pred)
    #end
    
    def get_cv_error(self):
        pred = self.predict('cv')
        return self.loss('cv', pred)
    #end
    
    def predict(self, kind = 'train'):
        self.forward_propagate(kind)
        return self.cache_a[-1]
    #end predict
    
# =============================================================================
#     # ToDo
#     def plot_decision_boundary(self):
#         x_min, x_max = self.train_samples[0, :].min() - 0.25, self.train_samples[0, :].max() + 0.25
#         y_min, y_max = self.train_labels[0, :].min() - 0.25, self.train_labels[0, :].max() + 0.25
#         # Draw a grid
#         h = 0.01
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#         # Predict the function value for the whole grid
#         Z = model(np.c_[xx.ravel(), yy.ravel()])
#     #end
# =============================================================================
    def get_approx_train_error(self):
        # forward propagate and calculate error
        # WRONG - REWRITE
        weights_copy_plus = np.copy(self.weights)
        bias_copy_plus = np.copy(self.bias)
        weights_copy_minus = np.copy(self.weights)
        bias_copy_minus = np.copy(self.bias)
        epsilon = 0.00001
        for i in weights_copy_plus:
            weights_copy_plus[i] = weights_copy_plus[i] + epsilon
            bias_copy_plus[i] = bias_copy_plus[i] + epsilon
            weights_copy_minus[i] = weights_copy_minus[i] - epsilon
            bias_copy_minus[i] = bias_copy_minus[i] - epsilon
        #end for
        
        self.forward_propagate(weights=weights_copy_plus, bias=bias_copy_plus)
        predict_plus = self.cache_a[-1]
        plus_err = self.loss('train', predict_plus)
        
        self.forward_propagate(weights=weights_copy_minus, bias=bias_copy_minus)
        predict_minus = self.cache_a[-1]
        minus_err = self.loss('train', predict_minus)
        
        approx_err = (plus_err - minus_err) / (2 * epsilon)
        return approx_err
    #end check_gradient
    
    def round(self, num=0):
        if (num-int(num))>=0.5:
            return math.ceil(num)
        else:
            return math.floor(num)
        #else
    #end 
#end NeuralNetwork

