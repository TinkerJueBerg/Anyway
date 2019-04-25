# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 12:10:49 2018

@author: 真夜绫也
"""


import numpy as np
from sklearn.model_selection import train_test_split
from  matplotlib import pyplot as  plt
from matplotlib import cm 
from matplotlib import axes
import pandas as pd

# coding=utf-8
def draw_heatmap(data,xlabels,ylabels):

    #cmap=cm.Blues    

    cmap=cm.get_cmap('rainbow',1000)
    figure=plt.figure(facecolor='w')
    ax=figure.add_subplot(1,1,1,position=[0.1,0.15,0.8,0.8])
    ax.set_yticks(range(len(ylabels)-4))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)-4))
    ax.set_xticklabels(xlabels)
    vmax=data[0][0]
    vmin=data[0][0]
    for i in data:
        for j in i:
            if j>vmax:
                vmax=j
            if j<vmin:
                vmin=j
    map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',vmin=vmin,vmax=vmax)
    cb=plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)
    plt.show()


a=np.random.rand(10,10)
xlabels=['A','B','C','D','E','F','G','H','I','J']
ylabels=['a','b','c','d','e','f','g','h','i','j']
draw_heatmap(a,xlabels,ylabels)  

"""
df = DataFrame({'key1':['a','a','b','b','a'],'key2':['one','two','one','two','one'],'data1':[1,2,3,4,5],'data2':[6,7,8,9,10]})
df1 = pd.pivot_table(df,index='key1',columns='key2',aggfunc='count')
print(df )
print(df1)
"""

