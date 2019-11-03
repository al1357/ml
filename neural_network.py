# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 23:56:35 2018

@author: al2357
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import float64, int
import math
import performance_measure as pm
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
    cache_dw = {}
    cache_db = {}
    # error cache iteration-error
    cache_train_error = []
    cache_cv_error = []
    cache_iterations = []
    cache_mb_train_error = []
    cache_mb_cv_error = []
    cache_mb_iterations = []
    # learing rate
    weights = {}
    bias = {}
    buffer = {}
    
    regularization = True
    reg_lambda = 0.00001
    dropout = False
    dropout_keep_prob = 0.8
    
    data_min = 0
    data_max = 0
    
    performance_measure = None
    last_layer_af = "softmax"
    
    early_stopping = False
    mini_batch_size_n = -1 # 2^n
    iterations = 10000
    max_iterations = 100000
    alpha = 0.1

    
    def __init__(self, data_loader, \
                     network_map=[], \
                     load_parameters=False):
        """
            training/test sets dimensions are n x m
        """
        np.random.seed(3)
        self.network_map = network_map
        self._dl = data_loader
        
        self.m['train'] = self._dl.get_x().shape[1]
        self.m['cv'] = self._dl.get_x('cv').shape[1]
        self.n = self._dl.get_x().shape[0]
        
        # n x m
        self.data_samples['train'] = self._dl.get_x()
        self.labels['train'] = self._dl.get_y()
        
        self.data_samples['cv'] = self._dl.get_x('cv')
        self.labels['cv'] = self._dl.get_y('cv')
        
        train_min = np.min(self.data_samples['train'])
        cv_min = np.min(self.data_samples['cv'])
        self.data_min = train_min if train_min < cv_min else cv_min
        train_max = np.max(self.data_samples['train'])
        cv_max = np.max(self.data_samples['cv'])
        self.data_max = train_max if train_max > cv_max else cv_max
        self.performance_measure = pm.performance_measure(self)
        
        # load saved parameters
        if load_parameters:
            self.load_parameters()
        else:   
            self.initialize_parameters('he')
        #end if
        
        #self.normalize()
        self.scale()
    #end
    
    def save_parameters(self):
        for l in self.weights:
            with open('parameters/weights_l'+str(l)+'.txt', 'w') as outfile:
                outfile.write('# Weights for layer {0}; shape {1}\n'.format(l, self.weights[l].shape))
                np.savetxt(outfile, self.weights[l])
            #end with
        #end for
        
        with open('parameters/bias.txt', 'w') as outfile:
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
                self.weights[i] = np.loadtxt('parameters/weights_l'+str(i)+'.txt')
                i = i + 1
            #end for
            bias = np.loadtxt('parameters/bias.txt')            
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
    
    def scale(self):
        '''data between [0, 1]'''
        self.data_samples['train'] = (self.data_samples['train'] - self.data_min) / (self.data_max - self.data_min)
        self.data_samples['cv'] = (self.data_samples['cv'] - self.data_min) / (self.data_max - self.data_min)
    #end scale
    
    def normalize(self):
        '''normal distribution; bell curve'''
        # calculate mean and variance only for training set; use them to normalize the rest
        mean = np.sum(self.data_samples['train'], axis=1, keepdims=True) / self.m['train']
        self.data_samples['train'] = (self.data_samples['train'] - mean) / (self.data_max - self.data_min)
        self.data_samples['cv'] = (self.data_samples['cv'] - mean) / (self.data_max - self.data_min)
    #end normalize
    
    def standarize(self):
        '''mean 0; variance 1'''   
        mean = np.sum(self.data_samples['train'], axis=1, keepdims=True) / self.m['train']
        # if variance is 0, e.g. feature is 0 for all samples, then Python returns divide by 0 warning
        variance = np.sum(self.data_samples['train']**2, axis=1, keepdims=True) / self.m['train']
        
        for ix in self.data_samples:
            self.data_samples[ix] = (self.data_samples[ix] - mean) / variance
            np.nan_to_num(self.data_samples[ix], False)
        #end for
    #end standarize
    
    def sigmoid(self, z):
        """Sigmoid function - multi-label 
        prediciton: sigmoid >= 0.5 result 1; sigmoid < 0.5 result 0;
        """
        return 1 / (1 + np.exp(-z))
    #end
    
    def d_sigmoid(self, z):
        """ derivative of the sigmoid function """
        zPrim = self.sigmoid(z)
        return zPrim*(1-zPrim)
    #end
    
    def softmax(self, z):
        """softmax - multi-class, single-label
        """
        t = np.exp(z - np.max(z))
        #t = np.exp(z)
        return t / np.sum(t, axis=0)
    #end
    
    def d_softmax(self, z):
        softmax = self.softmax(z)
        return softmax*(1-softmax)
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

    def forward_propagate(self, kind='train', weights=False, bias=False, data_in = [], dropout_in=True):
        if len(data_in) != 0:
            X = data_in
        elif kind == 'train' or kind == 'test' or kind == 'cv':
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
                # Last is softmax/sigmoid
                if self.last_layer_af == "softmax":
                    layer_a = self.softmax(layer_z)
                else:
                    layer_a = self.sigmoid(layer_z)
                #end if
            else:
                # Non-last are tanh/sigmoid
                layer_a = self.tanh(layer_z)
            #end if
            np.nan_to_num(layer_a, False)
            if(dropout_in and self.dropout and i < len(self.network_map)):
                # if hidden units and dropout then apply dropout
                dx = np.random.rand(layer_a.shape[0], layer_a.shape[1]) < self.dropout_keep_prob
                layer_a = np.multiply(layer_a, dx)
                layer_a /= self.dropout_keep_prob
            #end if dropout
            
            self.cache_a.append(layer_a)
        #end for
    #end forwardPropagate
    
    def back_propagate(self, kind='train', labels_in=[]):
        if kind != 'train' and kind != 'test' and kind != 'cv':
            return 0
        #end if
        if len(labels_in) != 0:
            labels = labels_in
        else:
            labels = self.labels[kind]
        #end if
        # e.g. range(2, 0) = [2, 1]
        n_depth = len(self.network_map)
        self.cache_dw = {}
        self.cache_db = {}
        prev_dz = []
        for i in range(n_depth, 0, -1):
            if i == n_depth:
                # last layer sigmoid / softmax
                dz = (self.cache_a[i] - labels)
            else:
                # i = 1; s1 contain n1 neurons; (n1, m) = (n1, n2) x (n2, m) * (n1, m)
                dz = np.dot(self.weights[i+1].T, prev_dz)*self.d_tanh(self.cache_z[i])
            prev_dz = dz
            self.cache_dw[i] = np.dot(dz, self.cache_a[i-1].T) / self.m[kind]
            self.cache_db[i] = np.sum(dz, axis=1, keepdims=True) / self.m[kind]
            # no dropout regularization
            if(self.regularization and not self.dropout):
                reg = (self.reg_lambda * self.weights[i]) / (2 * self.m[kind])
            else:
                reg = 0;
            self.weights[i] = self.weights[i] - self.alpha * (self.cache_dw[i] + reg)
            self.bias[i] = self.bias[i] - self.alpha * self.cache_db[i]
    #end backPropagate
    
    def learn(self, gradient_check = False):
        iterate = True
        cv_err_prev = -1
        cv_err = -1
        i = 0
        batch_i = 0
        if(self.mini_batch_size_n != -1):
            mb_size = np.power(2, self.mini_batch_size_n)
            batches_count = int(self.m['train'] / mb_size) + 1
        else:
            mb_size = 0
            batches_count = 0
        #end if            
        while iterate:
            # Early stopping condition
            i += 1
            if(self.early_stopping):
                cv_err_prev = cv_err
                cv_err = self.get_cv_error()
                if(self.max_iterations <= i or (i > 10 and cv_err_prev != -1 and cv_err_prev <= cv_err)):
                    iterate = False
                #end if
            elif(self.iterations <= i):
                iterate = False
            #end if
            
            if(self.mini_batch_size_n == -1):
                # Batch gd
                self.batch_gd(i)
            else:
                # Mini-batch gd
                batch_i = (i-1) * mb_size
                self.mini_batch_gd(batches_count, mb_size, batch_i)
            #end if
            
            # optional gradient check
            if gradient_check == True and (iterate == False or i == 1):
                grad_diffs = self.performance_measure.check_gradients()
                print("Gradient chack after %1d iterations: %10.10f" % (i, grad_diffs))
            #end if
            
        #end while
        
        self.performance_measure.performance('cv')
        if(self.mini_batch_size_n == -1):
            # batch gd
            print("Last train error: ", self.cache_train_error[-1])
        else:
            # mini-batch gd
            print("Last train error: ",self.get_train_error())
        #end if
        print("Last cv error: ", self.cache_cv_error[-1])
        plt.plot(self.cache_iterations, self.cache_train_error)
        plt.plot(self.cache_iterations, self.cache_cv_error)
        plt.legend(['train error', 'cv error'], loc='upper left')
        plt.title("Error vs iterations")
        plt.xlabel("iterations")
        plt.ylabel("error")
        plt.show()
        self.save_parameters()
    #end learn
    
    def batch_gd(self, i):
        self.forward_propagate()
        self.back_propagate()
    
        if i%100 == 0 or i == 1:
            training_error = self.get_train_error()
            print("Training error after %1d iterations: %10.10f" % (i, training_error))
            self.cache_train_error.append(training_error)
            self.cache_cv_error.append(self.get_cv_error())
            #self.cache_test_error.append(self.get_test_error())
            self.cache_iterations.append(i)
        #end if
    #end batch_gd
    
    def mini_batch_gd(self, batches_count, mb_size, batch_i):       
        for j in range(1, batches_count):
            batch_i += 1
            idx_start = mb_size * (j-1)
            idx_end = self.m['train'] if j == batches_count else mb_size * j
            data_mb = self.data_samples['train'][:, idx_start:idx_end]
            labels_mb = self.labels['train'][:, idx_start:idx_end]
            self.forward_propagate(data_in=data_mb)
            cache_a_temp = self.cache_a
            cache_z_temp = self.cache_z
            if batch_i%5 == 0 or batch_i == 1:
                # cost 
                train_error_mb = self.performance_measure.loss(kind='train', labels_in=labels_mb, m_in=data_mb.shape[1])
                self.cache_train_error.append(train_error_mb)
                self.cache_cv_error.append(self.get_cv_error())
                self.cache_iterations.append(batch_i)
            #end if
            self.cache_z = cache_z_temp
            self.cache_a = cache_a_temp
            self.back_propagate(labels_in=labels_mb)
        #end for
    #end mini_batch_gd
    
    def get_train_error(self):
        pred = self.predict('train')
        return self.performance_measure.loss('train', pred)
    
    def get_test_error(self):
        pred = self.predict('test')
        return self.performance_measure.loss('test', pred)
    #end
    
    def get_cv_error(self):
        pred = self.predict('cv')
        return self.performance_measure.loss('cv', pred)
    #end
    
    def predict(self, kind='train', custom_data=[], output="raw"):
        self.forward_propagate(kind, data_in=custom_data, dropout_in=False)
        if(output == "raw"):    
            return self.cache_a[-1]
        elif(output == "boolean"):
            return (self.cache_a[-1] >= 0.5)
    #end predict
    
    def round(self, num=0):
        if (num-int(num))>=0.5:
            return math.ceil(num)
        else:
            return math.floor(num)
        #else
    #end 
#end neural_network

