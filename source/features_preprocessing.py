'''
Created on Dec 25, 2013

@author: alessandro
'''

from sklearn import preprocessing
import numpy as np


class Scaler(object):
    
    
    def __init__(self, bias_and_variance_flag = True, log10_flag = False, log2_flag = False, log1p_flag = False, mask = None):
        
        self.bias_and_variance_flag = bias_and_variance_flag
        self.log10_flag = log10_flag
        self.log2_flag = log2_flag
        self.log1p_flag = log1p_flag
        self.mask = mask
        self.bias_and_variance_scaler = None
        
    
    def fit(self, X):
    
        if self.mask is None:
            masked_X = X
        else:
            masked_X = X[self.mask]
        
        if self.log10_flag:
            masked_X = np.log10(masked_X)
            
        if self.log2_flag:
            masked_X = np.log2(masked_X)
            
        if self.log1p_flag:
            masked_X = np.log1p(masked_X)
            
        if self.bias_and_variance_flag:
            self.bias_and_variance_scaler = preprocessing.StandardScaler().fit(masked_X)
            masked_X = self.bias_and_variance_scaler.transform(masked_X)
            
        return masked_X
    
    
    def transform(self, X):
        
        if self.mask is None:
            masked_X = X
        else:
            masked_X = X[self.mask]
        
        if self.log10_flag:
            masked_X = np.log10(masked_X)
            
        if self.log2_flag:
            masked_X = np.log2(masked_X)
            
        if self.log1p_flag:
            masked_X = np.log1p(masked_X)
            
        if self.bias_and_variance_flag:
            masked_X = self.bias_and_variance_scaler.transform(masked_X)
            
        return masked_X
            
        