import pandas as pd
import numpy as np
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

class data_loader:
    
    #cv and test sets
    sets_distribution = None
    
    x_train = None
    x_cv = None
    x_test = None
    
    y_train = None
    y_cv = None
    y_test = None
    
    x_min = None
    x_max = None
    
    def __init__(self, sets_distribution=[0, 0]):
        '''
        sets_distribution [cv, test]
        '''
        self.sets_distribution = sets_distribution
    #end __init__
    
    def get_x(self, kind="train", mod="raw"):
        '''
        kind cv|test
        returns n x m
        '''
        data = None
        if kind == "cv":
            data = self.x_cv
        elif kind == "test":
            data = self.x_test
        else:
            data = self.x_train
        #end if
        
        if mod == "scaled":
            return self.scale(data)
        else:
            return data
        #end if
    #end get_x
    
    def get_y(self, kind="train"):
        '''
        kind cv|test
        returns n x m
        '''
        if kind == "cv":
            return self.y_cv
        elif kind == "test":
            return self.y_test
        else:
            return self.y_train
    #end get_y
    
    def load_mnist(self, less_than = None, limit_to = None, one_hot = True, flatten = True):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        #concatenate
        x_all = np.concatenate((x_train, x_test))
        y_all = np.concatenate((y_train, y_test))
        m_train = x_all.shape[0]
        
        if(limit_to != None and limit_to <= m_train):
            x_all = x_all[0:limit_to, ...]
            y_all = y_all[0:limit_to]
            m_train = limit_to
        #end
        
        if(less_than != None):
            less_than_bin = y_all < less_than
            x_all = x_all[less_than_bin, ...]
            y_all = y_all[less_than_bin]
            m_train = x_all.shape[0]
        #end
        
        if(flatten == True):
            x_all = x_all.reshape(x_all.shape[0], x_all.shape[1]*x_all.shape[2])
        #end
            
        if(one_hot):
            y_all = to_categorical(y_all)
        #end
        
        self.x_min = np.min(x_all)
        self.x_max = np.max(x_all)
        
        # split into sets
        if self.sets_distribution[0] != 0 and self.sets_distribution[0] < 1:
            m_cv = math.floor(m_train * self.sets_distribution[0])
            self.x_cv = x_all[0:m_cv, ...].T
            self.y_cv = y_all[0:m_cv, ...].T
        else:
            m_cv = 0
            self.x_cv = np.array([])
            self.y_cv = np.array([])
        #end if    
            
        if self.sets_distribution[1] != 0 and self.sets_distribution[1] < 1:
            m_test = math.floor(m_train * self.sets_distribution[1])
            self.x_test = x_all[m_cv:(m_cv+m_test), ...].T
            self.y_test = y_all[m_cv:(m_cv+m_test), ...].T
        else:
            m_test = 0
            self.x_test = np.array([])
            self.y_test = np.array([])
        #end if    
              
        m_train -= (m_cv + m_test)
        self.x_train = x_all[(m_cv+m_test):, ...].T
        self.y_train = y_all[(m_cv+m_test):, ...].T
    #end load_mnist
    
    def load_iris(self, csv_path="data/iris.csv"):
        # load and prepare data
        #pandas.core.frame.DataFrame
        data = pd.read_csv(csv_path)
        
        # 2) encode categories into numbers - a: how to decode?
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        
        le.fit(data.Species.drop_duplicates()) # d_d() returns only unique values
        data.Species = le.transform(data.Species)
        
        # 3) split data into x and y; one-hot encode labels        
        np_data = np.array(data.sample(frac=1).reset_index(drop=True).iloc[:, :])
        x_all = np_data[:, 0:-1]
        y_all = to_categorical(np_data[:, -1])
        m_train = x_all.shape[0]
        
        self.x_min = np.min(x_all)
        self.x_max = np.max(x_all)

        # 4) split into sets
        if self.sets_distribution[0] != 0 and self.sets_distribution[0] < 1:
            m_cv = math.floor(m_train * self.sets_distribution[0])
            self.x_cv = x_all[0:m_cv, ...].T
            self.y_cv = y_all[0:m_cv, ...].T
        else:
            m_cv = 0
            self.x_cv = np.array([])
            self.y_cv = np.array([])
        #end if    
            
        if self.sets_distribution[1] != 0 and self.sets_distribution[1] < 1:
            m_test = math.floor(m_train * self.sets_distribution[1])
            self.x_test = x_all[m_cv:(m_cv+m_test), ...].T
            self.y_test = y_all[m_cv:(m_cv+m_test), ...].T
        else:
            m_test = 0
            self.x_test = np.array([])
            self.y_test = np.array([])
        #end if    
              
        m_train -= (m_cv + m_test)
        self.x_train = x_all[(m_cv+m_test):, ...].T
        self.y_train = y_all[(m_cv+m_test):, ...].T
    #end load_iris
    
    def load_cars(self, csv_path="data/121229-tauris_estate-p.csv"):
        data = pd.read_csv(csv_path)
        # shuffle
        data = data.sample(frac=1).reset_index(drop=True)
        prices = data.pop('price')
        x_all = np.array(data.iloc[:,:])
        y_all = np.array(prices.iloc[:])
        m_train = x_all.shape[0]
        y_all = y_all.reshape((m_train, 1))
        
        self.x_min = np.min(x_all)
        self.x_max = np.max(x_all)
        
        if self.sets_distribution[0] != 0 and self.sets_distribution[0] < 1:
            m_cv = math.floor(m_train * self.sets_distribution[0])
            self.x_cv = x_all[0:m_cv, ...].T
            self.y_cv = y_all[0:m_cv, ...].T
        else:
            m_cv = 0
            self.x_cv = np.array([])
            self.y_cv = np.array([])
        #end if

        if self.sets_distribution[1] != 0 and self.sets_distribution[1] < 1:
            m_test = math.floor(m_train * self.sets_distribution[1])
            self.x_test = self.x_all[m_cv:(m_cv+m_test), ...].T
            self.y_test = self.y_all[m_cv:(m_cv+m_test), ...].T
        else:
            m_test = 0
            self.x_test = np.array([])
            self.y_test = np.array([])
        #end if
        
        m_train -= (m_cv+m_test)
        self.x_train = x_all[(m_cv+m_test):, ...].T
        self.y_train = y_all[(m_cv+m_test):, ...].T
    #end load_cars
    
    def scale(self, data):
        return (data - self.x_min) / (self.x_max - self.x_min)
    #end scale
    
#end class data_loader

# =============================================================================
# Testing
#    
# dl = data_loader([0.2,0.3])
# dl.load_iris()
# x_train = dl.get_x()
# y_train = dl.get_y()
# x_cv = dl.get_x("cv")
# y_cv = dl.get_y("cv")
# x_test = dl.get_x("test")
# y_test = dl.get_y("test")
# =============================================================================
    
# =============================================================================
# # Data info
# # summarize dataset shape
# print('Train shape x, y: ', x_train.shape, y_train.shape)
# #print('Test shape x, y: ', (x_test.shape, y_test.shape))
# # summarize pixel values
# print('Train min, max, mean, std: ', x_train.min(), x_train.max(), x_train.mean(), x_train.std())
# #print('Test min, max, mean, std: ', x_test.min(), x_test.max(), x_test.mean(), x_test.std())
# =============================================================================
