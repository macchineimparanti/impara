'''
Created on Dec 25, 2013

@author: alessandro
'''
from scipy.sparse.linalg.eigen.arpack import eigsh
import numpy as np


class PCA(object):
    
    
    def __init__(self,  variance_retain = None, num_components = None):
    
        assert variance_retain is not None or num_components is not None
        if variance_retain is not None:
            assert 0<=variance_retain<=1.0
            
            self.variance_retain = variance_retain
            self.num_components = num_components
            
    
    def pca(self, data):
        
        # get dimensions
        num_data,dim = data.shape
        
        if dim>num_data:
            K = np.dot(data,data.T)
            eigen_values,eigen_vectors  = eigsh(K,k = np.linalg.matrix_rank(K)-1,which = 'LA')
            U = np.dot(data.T,eigen_vectors/np.sqrt(eigen_values))
            eigen_values, eigen_vectors = eigen_values[::-1]/(len(data)-1),U[:,::-1]      
        else:
            U,eigen_values,eigen_vectors = np.linalg.svd(data,full_matrices=False)
            eigen_vectors=eigen_vectors.T
            
        return eigen_vectors, eigen_values.cumsum(axis=0)/eigen_values.sum()
    
    
    def transform(self, data):
        
        assert hasattr(self,"pca_coeffs")
        
        return np.dot(data, self.pca_coeffs)
    
    
    def inv_transform(self, score):
        
        assert hasattr(self,"pca_coeffs")
        
        return np.dot(score, self.pca_coeffs.T) 
    
    
    def fit(self, data):
                
        # get dimensions
        num_data,dim = data.shape
            
        self.decorrelated_coeffs, self.latent = self.pca(data)
        
        #if variance retain is specified, it has greater priority than the number of components
        if self.variance_retain is not None:  
            mask_stride = self.latent <= self.variance_retain
            mask = np.tile(mask_stride, (dim,1))
            self.pca_coeffs = self.decorrelated_coeffs[mask].reshape(dim,-1)
        else:
            self.pca_coeffs = self.decorrelated_coeffs[:,:self.num_components]
            
        return self.pca_coeffs
