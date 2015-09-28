'''
Created on Dec 8, 2013

@author: alessandro
'''
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import pandas


def print_model_selection_results(results, C_list, gamma_list = None ):
    
    #if gamma list is None we expect a SVM linear classifier
    if gamma_list is None:
        rbf = False
        assert results.ndim == 1
    else:
        rbf = True
        assert results.ndim == 2
        
    C_list_indeces = []
    for C in C_list:
        C_list_indeces.append("C = {0}".format(C))

    if rbf:
        gamma_list_columns = []
        for gamma in gamma_list:
            gamma_list_columns.append("gamma = {0}".format(gamma))
            
    if not rbf:
        return pandas.DataFrame(results, index = C_list_indeces)
    else:
        return pandas.DataFrame(results, index = C_list_indeces, columns = gamma_list_columns)        
    

def plot_features(X):
    
    assert isinstance(X,np.ndarray)
    assert X.ndim == 2
    
    cols, rows = X.shape
    
    xaxis = np.tile( np.arange(rows), (cols,1))
    
    fig = plt.figure()
    fig.canvas.set_window_title('Features visualization.') 
    ax = fig.add_subplot(111)
    
    ax.plot(xaxis,X,'*')


def plot_2d( x, y, xlabel = "log10(C)" , ylabel = "accuracy", title = "Accuracy by C"):
        
        fig = plt.figure()
        fig.canvas.set_window_title('{0}'.format(title)) 
        ax = fig.add_subplot(111)
        
        ax_xlabel = ax.set_xlabel(xlabel)
        ax_ylabel = ax.set_ylabel(ylabel)
        
        x=np.log10(x)
        surf = ax.plot(x, y)
        ax.set_ylim(0, 1.01)


def plot_3d( x, y, z, xlabel = "log10(gamma)", ylabel ="log10(C)" , zlabel = "accuracy", title = "Accuracy by C and gamma"):
        
        fig = plt.figure()
        fig.canvas.set_window_title('{0}'.format(title)) 
        ax = fig.add_subplot(111, projection='3d')
        
        ax_xlabel = ax.set_xlabel(xlabel)
        ax_ylabel = ax.set_ylabel(ylabel)
        ax_zlabel = ax.set_zlabel(title)
        
        x1=np.log10(x)
        y1=np.log10(y)
        z1=z
        X1,Y1=np.meshgrid(np.array(x1), np.array(y1))
        surf = ax.plot_surface(X1, Y1, z1, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(-1.01, 1.01)
    
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
        fig.colorbar(surf, shrink=0.5, aspect=5)
        

def plot_confusion_metrics(self, cm):
             
        print "Confusion matrix:"   
        print cm
        
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


def plot_rfe_curve(gridscores, ylabel):

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel(ylabel)
    plt.plot(range(1, len(gridscores) + 1), gridscores)


def save_csv_submitted_labels(predicted_labels, filename):
       
        f = open(filename, "w")
        len_labels = predicted_labels.shape[0]
        f.write("Id,Solution\n")
        for i in xrange(len_labels):
            f.write("{0},{1}\n".format(i+1,int(predicted_labels[i])))
        f.close()
        

def save_sf_features(sf, training_set, test_set, path):
    
    container = dict()
    container["sf"] = sf
    container["training_set"] = training_set
    container["test_set"] = test_set
    
    if not os.path.exists(os.path.dirname(path)):
        raise Exception("The specified directory for saving sparse filtering features does not exist.")
    
    f = open(path,"w")
    pickle.dump(container, f)
    f.close()
    
    
def load_sf_features(path):
    
    f = open(path,"r")
    container = pickle.load(f)
    f.close()
    
    return container["sf"], container["training_set"], container["test_set"]
