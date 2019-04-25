# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:47:28 2019

@author: 真夜绫也
"""

import numpy as np
from sklearn.model_selection import train_test_split

dataframe = np.array([[0,'q','left'],[1,'w','right'],[2,'e','left'],[3,'r','right'],[4,'t','right']])
#First
"""
d = dataframe.T
dx = d[:2].T
dy = d[2]
print(dx,dy)

X_train,X_test,y_train,y_test = train_test_split(dx,dy,test_size=0.4,random_state=0)
print('a\n',X_train,'\nb\n',X_test,'\nc\n',y_train,'\nd\n',y_test)
"""

#second
"""
dy=[]
for i in range (5):
    dy.append(dataframe[i,2])
dx=[]
for i in range (5):
    for j in range(2):
        dx.append(dataframe[i,j])
dx = np.array(dx)
dx = dx.reshape(5,2)

X_train,X_test,y_train,y_test = train_test_split(dx,dy,test_size=0.4,random_state=0)
print('a\n',X_train,'\nb\n',X_test,'\nc\n',y_train,'\nd\n',y_test)
"""

x = range(4)
print(x)