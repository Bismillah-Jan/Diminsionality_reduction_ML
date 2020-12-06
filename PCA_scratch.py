# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:34:35 2020

@author: bismillah.jan
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

# x1, x2, x3

data=np.array([[1,1,3],[2,2,1],[2,-1,1],[3,1,-1],[1,-2,2], 
               [3,3,4],[4,4,3], [3,4,5], [5,1,3],[5,3,3]])
labels=np.array([-1,-1,-1,-1,-1, 1, 1, 1, 1, 1])
nf=data.shape[1]

"""
How covariance is computed?
mi=np.mean(data[:,0])
mj=np.mean(data[:,1])
mk=np.mean(data[:,2])
cov(xi,xj)= sum{square(  (xi- mean(xi))*(xj-mean(xj))  )}

data[:,0]==first feature, and so on.

n=data.shape[0]-1
COV(data[:,i], data[:,j])=np.sum((data[:,i]-mi)*(data[:,j]-mj))/n
"""

def COV(i,j, mi, mj):
    n=data.shape[0]-1
    return np.round(np.sum((data[:,i]-mi)*(data[:,j]-mj))/n,2)

def printCov():
    for i in range(0,3):
        for j in range(0,3):
            print COV(i, j, np.mean(data[:,i]), np.mean(data[:, j])),', ',
        print


def plot_data(data, dim):
    fig = plt.figure()
    if dim==3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax=fig.add_subplot(111)
  
    i=0;
    for l in labels:
        if l==-1:
            col='r'
        else:
            col='b'
        if dim==3:
            ax.scatter(data[i,0],data[i,1], data[i,2], marker='o',color=col)
            ax.text(data[i,0],data[i,1],data[i,2],  '%s' % (str(i+1)), size=15, zorder=1, color=col) 

            i+=1
        else:
            ax.scatter(data[i,0],data[i,1], marker='o', color=col)
            ax.text(data[i,0],data[i,1],  '%s' % (str(i+1)), size=15, zorder=1, color=col) 
            i+=1
    if dim==3:            
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    else:       
       
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        
    plt.show()





covM=np.round(np.cov(data.T),2)
eignvalues, eignVector=np.linalg.eig(covM)

e_index_sorted=np.argsort(eignvalues) #return sorted eignvalues in increasing order

projection_Matrix=eignVector[:, e_index_sorted[1:]] #discarded zero-th one
#we have discarded the first eignVector corresponeint to lowest eignvalues

"""
Now it's the time to transform our data with reduced dimension using projection matrix
"""

new_data=np.dot(data,projection_Matrix) #new_data is our reduced features dataset
#let's draw it.

plot_data(new_data, 2)
plot_data(data, 3)
