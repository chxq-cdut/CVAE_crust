# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 08:22:23 2019

@author: Xianqoing Cheng
"""


import numpy as np 

RegionAll=['Sedu','Sedm','Sedl','Crsu','Crsm','Crsl','Sed','Crs']  #要反演的区域
reg=[1,2,5]
for k in np.arange(0,3):
    print('Region=',RegionAll[reg[k]])
    data_Y_ini_tmp = np.loadtxt('trainY.txt')
    data_Y_ini = data_Y_ini_tmp[data_Y_ini_tmp[:,reg[k]].argsort()] #按照第reg+1列对行排序
        
    data_X_ini_tmp = np.loadtxt('trainX.txt')
    
    data_X_ini = data_X_ini_tmp[data_Y_ini_tmp[:,reg[k]].argsort()]  #根据 Y的排序，相应排序X
    
    data_Y_ini_sed=data_Y_ini[:,reg[k]:reg[k]+1] #针对沉积层厚度的处理
    
    if(reg[k]<=2):
        group = np.arange(min(data_Y_ini_sed).astype(int),max(data_Y_ini_sed).astype(int)+2,1)  #[0,1,2,3,4,10]
    else:
        group = np.arange(min(data_Y_ini_sed).astype(int),max(data_Y_ini_sed).astype(int)+10,5) #[5,10,15,20,25,35]   #地壳
       
    [x1,x2]=np.histogram(data_Y_ini_sed,bins=group) #计算原始数据在每个区间的分布,kk2-原始数据在每个区间的个数
    array_num1=x1 
    if(k==0):
        array_num1[-1]=0.33*len(data_Y_ini_sed)
    elif(k==1):
        array_num1[-1]=0.354*len(data_Y_ini_sed) 
    else:
        array_num1[-1]=0.19*len(data_Y_ini_sed)
    array_num=array_num1.astype(int)
    
    
    kk3=np.append(0,np.cumsum(x1)) #每个区间数据在原始文件中的起始点位置
    for i in range(len(kk3)-1):
        
        yy=data_Y_ini[kk3[i]:kk3[i+1],:]    # 每个区间的数据 
        xx=data_X_ini[kk3[i]:kk3[i+1],:]
        index = [j for j in range(len(yy))] # 每个区间数组从0开始编号
        np.random.shuffle(index)           #每个区间数据打乱
        index1=np.random.choice(index, size=array_num[i], replace=True, p=None) #从第i个区间随机挑选array_num[i]个数据的位置
        
        yy1=yy[index1[0:array_num[i]],:]    #从第i个取间随机挑选array_num[i]个数据
        xx1=xx[index1[0:array_num[i]],:]
        
        if i==0:
            tmpY=yy1 
            tmpX=xx1
        else:
            tmpY=np.concatenate((tmpY,yy1),axis=0)
            tmpX=np.concatenate((tmpX,xx1),axis=0)
        
    data_Y=tmpY 
    data_X=tmpX 
    
    
    
    [x3,x4]=np.histogram(data_Y[:,reg[k]:reg[k]+1],bins=group,weights=None,density=False)
    kk6=np.append(0,np.cumsum(x3))
    
    
    np.savetxt('trainY_Equ'+str(reg[k])+'.txt',data_Y,fmt=" %8.4f"*data_Y.shape[1])
    np.savetxt('trainX_Equ'+str(reg[k])+'.txt',data_X,fmt=" %8.4f"*data_X.shape[1])


