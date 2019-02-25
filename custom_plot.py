# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:54:11 2019

@author: al2357
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import float64, int
import math

class custom_plot:
    data = None
    def __init__(self, data):
        #constructor
        self.data = data
        
    def plot_iris(self):
        #for each Species ,let's check what is petal and sepal distibutuon
        setosa = self.data[self.data['Species'] == 'Iris-setosa']
        versicolor = self.data[self.data['Species'] == 'Iris-versicolor']
        virginica = self.data[self.data['Species'] == 'Iris-virginica']
        
        plt.figure()
        fig,ax=plt.subplots(2,2,figsize=(21, 10))
        
        # scatter
        setosa.plot(x="SepalLengthCm", y="SepalWidthCm", kind="scatter",ax=ax[0][0],label='setosa',color='r')
        versicolor.plot(x="SepalLengthCm",y="SepalWidthCm",kind="scatter",ax=ax[0][0],label='versicolor',color='b')
        virginica.plot(x="SepalLengthCm", y="SepalWidthCm", kind="scatter", ax=ax[0][0], label='virginica', color='g')
        
        setosa.plot(x="PetalLengthCm", y="PetalWidthCm", kind="scatter",ax=ax[0][1],label='setosa',color='r')
        versicolor.plot(x="PetalLengthCm",y="PetalWidthCm",kind="scatter",ax=ax[0][1],label='versicolor',color='b')
        virginica.plot(x="PetalLengthCm", y="PetalWidthCm", kind="scatter", ax=ax[0][1], label='virginica', color='g')
        
        setosa.plot(x="PetalLengthCm", y="SepalLengthCm", kind="scatter",ax=ax[1][0],label='setosa',color='r')
        versicolor.plot(x="PetalLengthCm",y="SepalLengthCm",kind="scatter",ax=ax[1][0],label='versicolor',color='b')
        virginica.plot(x="PetalLengthCm", y="SepalLengthCm", kind="scatter", ax=ax[1][0], label='virginica', color='g')
        
        # histogram
        setosa["SepalLengthCm"].plot(kind="hist", ax=ax[1][1],label="setosa",color ='r',fontsize=10)
        
        ax[0][0].set(title='Sepal comparasion ', ylabel='sepal-width')
        ax[0][1].set(title='Petal Comparasion',  ylabel='petal-width')
        ax[1][0].set(title="Petal/Sepal length comparison", ylabel='sepal-length', xlabel='petal-length')
        ax[1][1].set(title='SepalLengthCm')
        ax[0][0].legend()
        ax[0][1].legend()
        ax[1][0].legend()
        ax[1][1].legend()
        
        plt.show()
        
        # box plot
        plt.boxplot(setosa["SepalLengthCm"])
        plt.yticks(range(4, 6))
        plt.ylabel("Value")
        plt.show()