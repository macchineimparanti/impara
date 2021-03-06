'''
Created on Jan 28, 2014

@author: alessandro
'''
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from autodiff import optimize
from scipy import stats

def soft_absolute(u):
    epsilon = 1e-8
    return np.sqrt(u*u + epsilon)
      
def logistic(u):
    return 1. / (1. + np.exp(-u))

class SparseFilterLayer(BaseEstimator, TransformerMixin):
      
    def __init__(self, n_features = 200, n_iterations = 300, activate=soft_absolute):
        self.epsilon = 1e-8,
        self.n_features = n_features
        self.n_iterations = n_iterations
        self.activate = activate
    
    def fit(self, X, y = None):
        n_samples, n_dim = X.shape
        W = np.random.randn(n_dim, self.n_features)
        b = np.random.randn(self.n_features)
        obj_fn = self.get_objective_fn(X)
        self.W_, self.b_ = optimize.fmin_l_bfgs_b(obj_fn, (W, b),
                                                  iprint = 1,
                                                  maxfun = self.n_iterations)
        return self

    def get_objective_fn(self, X):
        
        def _objective_fn(W, b):
            
            Y = self.activate(np.dot(X, W) + b)
            Y = Y / np.sqrt(np.sum(Y*Y, axis = 0) + self.epsilon)
            Y = Y / np.sqrt(np.sum(Y*Y, axis = 1)[:, np.newaxis] + self.epsilon)
            
            return np.sum(Y)
        
        return _objective_fn

    def transform(self, X):
        
        Y = self.activate(np.dot(X, self.W_) + self.b_)
        Y = Y / np.sqrt(np.sum(Y*Y, axis=0) + self.epsilon)
        Y = Y / np.sqrt(np.sum(Y*Y, axis=1)[:, np.newaxis] + self.epsilon)
        
        return Y
    
class SparseFilter(object):
    
    def __init__(self, n_layers = 1, n_iterations = 1000, n_features = 120, layers_list = None):
        self.n_layers = n_layers
        self.n_iterations = n_iterations
        self.n_features = n_features
        
        if layers_list is not None:
            self.layers_list = layers_list
            self.n_layers = len(layers_list)
            self.n_features = int(layers_list[0].n_features)
        
    def fit(self, X):
        
        self.layers_list = []
        data = X
        for i in xrange(self.n_layers):
            layer = SparseFilterLayer(n_features = self.n_features,
                                   n_iterations = self.n_iterations)
            layer.fit(data)
            data = layer.transform(data)
            self.layers_list.append(layer)
            
    def transform(self, X):
        
        feat = None
        data = X
        for layer in self.layers_list:
            data = layer.transform(data)
            if feat is None:
                feat = data
            else:
                feat = np.hstack((feat, data))
                
        return feat
    
    def layers_ripartition(self):
        
        assert isinstance(self.layers_list, list)
        
        layers_ripartition_list = []
        
        for i in xrange(len(self.layers_list)):
            sub_sf = SparseFilter(layers_list=self.layers_list[:i+1])
            layers_ripartition_list.append(sub_sf)
            
        return layers_ripartition_list
    
    def get_layers_counter(self):
        
        return self.n_layers
    
    def get_features_counter(self):
        
        return self.n_features