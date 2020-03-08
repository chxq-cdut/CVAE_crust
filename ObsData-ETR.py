# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 08:22:23 2019

@author: Xianqoing Cheng
"""
# # =========观测数据多输出回归================ 根据ExtraTreesRegressor对Obs插值(把经纬度不一致的周期文件插值成经纬度范围一致的文件) ========================

# In[1]  
import numpy as np

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor,ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.metrics import explained_variance_score
from numpy.random import seed
seed(1)
# In[2]
path='China_2015_disp_v1.0\\'
All_File=np.loadtxt(path+'Filename.dat',dtype=np.str)
print(All_File.size)
new=np.loadtxt(path+'SameRange_Longmen.dat')

RegFun=[DecisionTreeRegressor(),LinearRegression(),SVR(),KNeighborsRegressor(),RandomForestRegressor(n_estimators=20),AdaBoostRegressor(n_estimators=50),\
        GradientBoostingRegressor(n_estimators=100),BaggingRegressor(),ExtraTreeRegressor(),KernelRidge(),MLPRegressor(),ExtraTreesRegressor()]
RegFunName=['DecisionTreeRegressor','LinearRegression','SVR','KNeighborsRegressor','RandomForestRegressor','AdaBoostRegressor',\
            'GradientBoostingRegressor','BaggingRegressor','ExtraTreeRegressor','KernelRidge','MLPRegressor','ExtraTreesRegressor']
prd=np.loadtxt(path+'Obs_Period.dat')
prd=prd.reshape(1,-1)
prd=np.insert(prd,[0,0],[0,0])
prd=prd.reshape(1,prd.size)

for j in range(0,12):
    for i in range(0,All_File.size):
    # 2 分割训练数据和测试数据
    # 随机采样25%作为测试 75%作为训练
        All_X=np.loadtxt(path+All_File[i])
        x_train, x_test, y_train, y_test = train_test_split(All_X[:,0:2], All_X[:,2:All_X.shape[1]], test_size=0.15, random_state=33)
        y_train= y_train[:,0:1]
        y_test=  y_test[:,0:1] 
        Predict_X =np.concatenate((x_train,y_train),axis=1)
        X=Predict_X[:,0:2]
        y=Predict_X[:,2:Predict_X .shape[1]]
    
      
    #极端随机森林回归
        regfun = RegFun[j]
        reg =MultiOutputRegressor(regfun)
        y_predict= reg.fit(x_train, y_train).predict(x_test) 
        newy = reg.fit(x_train, y_train).predict(new)  
            
       
        # 极端随机森林回归模型评估        
        print()
        print('FileName=',All_File[i])
        print(RegFunName[j]+"_mean_squared_error：",mean_squared_error(y_test, y_predict))
        print(RegFunName[j]+"_mean_absolute_error：",mean_absolute_error(y_test, y_predict))
        print(RegFunName[j]+"_explained_variance_score：",explained_variance_score(y_test, y_predict))
        print(RegFunName[j]+"_r2_score：",r2_score(y_test, y_predict))     
        
        
        
        if i==0:
                resy=newy         
        else:
                resy=np.concatenate((resy,newy),axis=1)        
            
    res = np.concatenate((new[:,0:1],new[:,1:2],resy),axis=1)
    
    res2=np.concatenate((prd,res),axis=0)
    #np.savetxt(path+'China_2015_disp_v1.0_201908\\ETR\\Obs_Latlong_ETR_'+RegFunName[j]+'.txt',res2,fmt="%6.2f %6.2f "+"%8.4f "*38 )
    if(RegFunName[j]=='ExtraTreesRegressor'):
        np.savetxt('Obs_'+RegFunName[j]+'.txt',resy,fmt="%8.4f "*38 )
  