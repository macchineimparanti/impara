'''
Created on Jan 28, 2014

@author: alessandro
'''
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class FeaturesSelectionRandomForests(object):
    
    
    def __init__(self, n_estimators = 100, feature_importance_th = 0.005):
        
        self.n_estimators = n_estimators
        self.feature_importance_th = feature_importance_th
        
            
    def fit(self, X, y, n_estimators = None, feature_importance_th = None):
        
        if n_estimators is not None:
            assert isinstance(n_estimators,(int,long,float))
            self.n_estimators = n_estimators
        if feature_importance_th is not None:
            assert isinstance(feature_importance_th,(int,long,float))
            self.feature_importance_th = feature_importance_th
        
        #filter features by forest model
        self.trees = ExtraTreesClassifier(n_estimators=100, compute_importances=True)
        self.trees.fit(X, y)
        self.features_mask = np.where(self.trees.feature_importances_ > 0.005)[0]

    
    def plot_features_importance(self):
        
        pd.DataFrame(self.trees.feature_importances_).plot(kind='bar')
        plt.show()
        
    
    def transform(self, X):

        assert hasattr(self,"features_mask")

        return X[:, self.features_mask]