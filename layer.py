# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 02:23:59 2024

@author: gaura
"""

import numpy as np

class Linear:
    def __init__(self, units):
        
        self.units = units
        self.initialized = False
        
    def __call__(self, x):
        
        if not self.initialized:
            self.w = np.random.randn(self.input.shape[-1], self.units)
            self.b = np.random.randn(self.units)
            self.initialized = True
            
        return self.input @ self.w + self.b
    
x = np.array([[0,1]])
layer = Linear(5)
print(layer(x))



class Sigmoid:
    def __call__(self, x):
        return 1/ (1 + np.exp(-x))
    
class Relu:
    def __call__(self, x):
        return np.maximum(0, x)
    
class Softmax:
    def __call__(self, x):
        return np.exp(x) / np.sum(np.exp(x))
    
class Tanh:
    def __call__(self, x):
        return np.tanh(x)
    

class Model:
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, x):
        output = x
        for layer in self.layers:
            output = layer(x)
            
        return output
    
