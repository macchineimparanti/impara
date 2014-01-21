'''
Created on Dec 8, 2013

@author: alessandro
'''
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


def plot_features(X):
    
    assert isinstance(X,np.ndarray)
    assert X.ndim == 2
    
    cols, rows = X.shape
    
    xaxis = np.tile( np.arange(rows), (cols,1))
    
    fig = plt.figure()
    fig.canvas.set_window_title('Features visualization.') 
    ax = fig.add_subplot(111)
    
    ax.plot(xaxis,X,'*')


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
