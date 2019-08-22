# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 21:11:04 2019

@author: al2357
"""

import neural_network as nn
import numpy as np

class test_nn:
    train_samples = None
    train_labels_bin = None
    def __init__(self, train_samples_in, train_labels_bin_in):
        self.train_samples = train_samples_in
        self.train_labels_bin = train_labels_bin_in
    
        
    def test1(self):
        '''Run nn with tanh and sigmoid and compare the results'''
        # hidden and output layers
        network_structure = [10, 5, 3]
        
        # train, cv, test sets
        tct = [0.6, 0.2, 0.2]
        
        my_data_in = np.array([[5, 3, 1, 0.1], [7, 1, 7, 1]]).T
        
        nn1 = nn.neural_network(network_structure, \
                                self.train_samples, \
                                self.train_labels_bin, \
                                tct, \
                                load_parameters=True)

        # nn1.check_gradients()
        w_derv = nn1.get_unrolled_derivatives()
        
        
        # nn1 predict
# =============================================================================
#         pr = nn1.predict(custom_data=my_data_in)
#         return pr
# =============================================================================
        
        
# =============================================================================
#         nn1.learn()
#         train_err = nn1.get_train_error();
#         print("Training error: "+str(train_err))
#         cv_err = nn1.get_cv_error()
#         print("CV error: "+str(cv_err))
#         test_err = nn1.get_test_error()
#         print("Test error: "+str(test_err))
# =============================================================================
    