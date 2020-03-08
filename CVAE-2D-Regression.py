# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 08:22:23 2019

@author: Xianqoing Cheng
"""

# In[2]

from keras import backend as K
from keras.models import Model #泛型模型  
from keras.layers import Dense, Input,BatchNormalization,Lambda 
from keras.optimizers import Adam,SGD,Adadelta
from keras.utils.vis_utils import plot_model
import numpy as np  
import h5py  
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
import matplotlib.pyplot as plt  
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler
from keras.models import load_model
from numpy.random import seed
from keras.losses import mse, binary_crossentropy
seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2) 
from keras.callbacks import LearningRateScheduler

# In[3]

Nump_ini=21
Numg_ini=17 
Numd_ini=9  
Numvp_ini=9   
Numvs_ini=9 
Numro_ini=9 
              
RegionAll=['Sedu','Sedm','Sedl','Crsu','Crsm','Crsl','Sed','Crs']  #name of layers
reg=0
if(reg!=6 and reg!=7):
    dcol=np.arange(reg,reg+1)
    region=RegionAll[reg]
    Numd=dcol.size                 
    vpcol=Numd_ini+dcol
    vscol=Numd_ini+Numvp_ini+dcol
    rocol=Numd_ini+Numvp_ini+Numvs_ini+dcol
     
     # network parameters
    
    intermediate_dim = 10
    batch_size =  9600 
    latent_dim = 4 
    epochs = 50
    layer=1
    
    LearningRate=1e-2 
    
    
    lumda=1e-6
    lumda2=1e-6 
    noise_factor =0.1 
    
    
    dr=0.3  #dropout rate (%)
    step=4 
    
    ActFun= ['relu','tanh','sigmoid','softmax','linear','softplus','softsign','hard_sigmoid']
    act=ActFun[0]
    DataProFun=[MinMaxScaler,StandardScaler,MaxAbsScaler]  
      
    DataPro= DataProFun[0]
    
    def MyLoss(y_true, y_pred, e=0.1): 
        return K.maximum(K.square(y_pred - y_true),0)
    def huber_loss(y_true, y_pred,delta=0.1):
        error = y_true - y_pred
        cond  = tf.keras.backend.abs(error) < delta
        squared_loss = 0.5 * tf.keras.backend.square(error)
        linear_loss  = delta * (tf.keras.backend.abs(error) - 0.5 * delta)
        return tf.where(cond, squared_loss, linear_loss)   
    def quan(y_true, y_pred, theta=0.5): 
        error = y_true-y_pred 
        loss = K.mean(K.maximum(theta*error, (theta-1)*error), axis=-1)    
        return loss 
    lossall=[MyLoss,quan,huber_loss,'cosine','poisson','kld','logcosh','binary_crossentropy','hinge','squared_hinge','msle' ,'mape','mae','mse']
    lossfun1=lossall[-1] 
    lossfun2=lossall[1]
    
    def my_init(shape, name=None):
        value = np.random.random(shape)
        return K.variable(value, name=name)
    InitFun=[my_init,'Initializer','Zeros','Ones','Constant','RandomNormal','RandomUniform','TruncatedNormal','VarianceScaling',\
             'Orthogonal','Identity','lecun_uniform','glorot_uniform','he_normal','lecun_normal','glorot_normal','he_uniform']         
    Kernini=InitFun[-5]
    Biasini=InitFun[3]
    
    adam = Adam(lr=LearningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-8,decay=0)               
    sgd=SGD(lr=LearningRate, momentum=0.0, decay=0, nesterov=False)
    adadelta=Adadelta(lr=LearningRate, rho=0.95, epsilon=1e-010)
    OptiFunAll=[adam,sgd,'RMSprop','Adagrad',adadelta,'Adamax','Nadam',]
    OptiFun=OptiFunAll[0]
    
    
    
    # In[4]
   
    
    period =np.array([8,10,12,14,16,18,20,22,24,26,28,30,32,35,40,45,50,55,60,65,70])
    pcol=np.arange(0,21) # phase Vel. 0~21，group Vel.21~38
    gcol=np.arange(21+15,38+15)
    gcol_pre=np.arange(21,38)
    
    dispNo=2   #1-Phase vel. 2-group vel.
    
    Nump=pcol.size  
    Numg=gcol.size
    
    if(reg==0 or reg==3 or reg==4):
        iniFile=0
    elif(reg==1):  #1,2,5
        iniFile=1
    elif(reg==2):
        iniFile=2
    else:
        iniFile=3
    
    iniXFile=['trainX.txt','trainX_Equ1.txt','trainX_Equ2.txt','trainX_Equ5.txt']
    print('reg=',reg,'   iniFile=',iniFile,'  iniXFile=',iniXFile[iniFile])
    data_X_ini = np.loadtxt(iniXFile[iniFile], unpack=False)
    data_X_ini_noisy=data_X_ini + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data_X_ini.shape)
    
    iniPreXFile=['Obs_ExtraTreesRegressor.txt']
    Predict_X_ini = np.loadtxt(iniPreXFile[0], unpack=False) 
    
    
    ################## Normalized X&Y  #######################
    data_X1_ini=data_X_ini[:,pcol] 
    data_X1_noisy_ini=data_X_ini_noisy[:,pcol]
    Predict_X1_ini=Predict_X_ini[:,pcol] 
    
    Predict_X1_temp=Predict_X1_ini   
    cmin=Predict_X1_ini.min()   
    cmax=Predict_X1_ini.max()  
    
    
    
    Len_data_X1=(data_X1_ini.shape)[0]   
    Len_Predict_X1=(Predict_X1_ini.shape)[0]
    data_X1_all=np.concatenate((data_X1_ini,data_X1_noisy_ini,Predict_X1_ini),axis=0)
    data_X1_all=data_X1_all.reshape((Len_data_X1*2+Len_Predict_X1)*Nump,1)
    
    ss_x1 = DataPro()
    data_X1_all=ss_x1.fit_transform(data_X1_all)
    data_X1_all=data_X1_all.reshape(Len_data_X1*2+Len_Predict_X1,Nump)
    data_X1=data_X1_all[0:Len_data_X1,:]
    data_X1_noisy=data_X1_all[Len_data_X1:Len_data_X1*2,:]
    Predict_X1=data_X1_all[Len_data_X1*2:Len_data_X1*2+Len_Predict_X1,:]
    
    data_X2_ini=data_X_ini[:,gcol] 
    data_X2_noisy_ini=data_X_ini_noisy[:,gcol]
    Predict_X2_ini=Predict_X_ini[:,gcol_pre] 
    Predict_X2_temp=Predict_X2_ini  
    umin=Predict_X2_ini.min()  
    umax=Predict_X2_ini.max()  
    
    Len_data_X2=(data_X2_ini.shape)[0]   
    Len_Predict_X2=(Predict_X2_ini.shape)[0]
    data_X2_all=np.concatenate((data_X2_ini,data_X2_noisy_ini,Predict_X2_ini),axis=0)
    data_X2_all=data_X2_all.reshape((Len_data_X2*2+Len_Predict_X2)*Numg,1)
    
    ss_x2 = DataPro()
    data_X2_all=ss_x2.fit_transform(data_X2_all)
    data_X2_all=data_X2_all.reshape(Len_data_X2*2+Len_Predict_X2,Numg)
    data_X2=data_X2_all[0:Len_data_X2,:]
    data_X2_noisy=data_X2_all[Len_data_X2:Len_data_X2*2,:]
    Predict_X2=data_X2_all[Len_data_X2*2:Len_data_X2*2+Len_Predict_X2,:]
    
   
    
    if(dispNo==2):
        data_X=data_X2           
        data_X_noisy=data_X2_noisy
        Predict_X=Predict_X2
    else:
        data_X=data_X1          
        data_X_noisy=data_X1_noisy
        Predict_X=Predict_X1 
    
    iniYFile=['trainY.txt','trainY_Equ1.txt','trainY_Equ2.txt','trainY_Equ5.txt','UM_vs_trainY_CPST_201909_th_Equ.txt']
    data_Y_ini = np.loadtxt(iniYFile[iniFile]) 
    print('reg=',reg,'   iniFile=',iniFile,'  iniXFile=',iniYFile[iniFile])
    data_Y1_ini=data_Y_ini[:,dcol]  
    temp=data_Y1_ini
    
    data_Y2_ini=data_Y_ini[:,vpcol]
    data_Y3_ini=data_Y_ini[:,vscol]
    data_Y4_ini=data_Y_ini[:,rocol]
    Len_data_Y2=(data_Y2_ini.shape)[0]
    Len_data_Y3=(data_Y3_ini.shape)[0]
    Len_data_Y4=(data_Y4_ini.shape)[0]
    
    # Discontinuty
    Len_data_Y1=(data_Y1_ini.shape)[0] 
    data_Y1_all=data_Y1_ini.reshape(Len_data_Y1*dcol.size,1)   
    ss_y1=DataPro()
    data_Y1_all = ss_y1.fit_transform(data_Y1_ini)
    data_Y1=data_Y1_all.reshape(Len_data_Y1,dcol.size)
    
    
    #vp
    Len_data_Y2=(data_Y2_ini.shape)[0] 
    data_Y2_all=data_Y2_ini.reshape(Len_data_Y2*dcol.size,1)   
    
    ss_y2=DataPro()
    data_Y2_all = ss_y2.fit_transform(data_Y2_ini)
    data_Y2=data_Y2_all.reshape(Len_data_Y2,dcol.size)
    
    #vs
    Len_data_Y3=(data_Y3_ini.shape)[0] 
    data_Y3_all=data_Y3_ini.reshape(Len_data_Y3*dcol.size,1)   
    ss_y3=DataPro()
    data_Y3_all = ss_y3.fit_transform(data_Y3_ini)
    data_Y3=data_Y3_all.reshape(Len_data_Y3,dcol.size)
    
    #ro
    Len_data_Y4=(data_Y4_ini.shape)[0] 
    data_Y4_all=data_Y4_ini.reshape(Len_data_Y4*dcol.size,1) 
    ss_y4=DataPro()
    data_Y4_all = ss_y4.fit_transform(data_Y4_ini)
    data_Y4=data_Y4_all.reshape(Len_data_Y4,dcol.size)   
    
    data_Y=np.concatenate((data_Y1,data_Y2,data_Y3,data_Y4),axis=1)
    
    
    # In[5]
    
    encod_dim_semi=Nump
    number=1
    for num in range(1,1+number):     
       # In[5]
        index = [i for i in range(len(data_X))] 
        np.random.shuffle(index)
        data_X = data_X[index]
        data_Y = data_Y[index]
        data_X_noisy=data_X_noisy[index]
        
        Train_X, Test_X, Train_Y, Test_Y = train_test_split(data_X, data_Y, test_size=0.05, random_state=33)
        Train_X_noisy, Test_X_noisy, Train_Y, Test_Y = train_test_split(data_X_noisy, data_Y, test_size=0.05, random_state=33)
        
        input_dim= data_X.shape[1]  # 98
        encoding_dim = data_Y.shape[1]   # discountinity h vs 3   
      
   
        def sampling(args):
            """Reparameterization trick by sampling from an isotropic unit Gaussian.
            # Arguments
            args (tensor): mean and log of variance of Q(z|X)
            # Returns
            z (tensor): sampled latent vector
            """
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            # by default, random_normal has mean = 0 and std = 1.0
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
       
        original_dim=input_dim
        input_shape = (original_dim, )
        encoder_input = Input(shape=(input_dim,)) 
      
       # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
         
        x = Dense(6,activation='relu')(inputs)    # 50 40 25 10 6 #20 18 14 10 6
        x = BatchNormalization()(x)
        x = Dense(10,activation='relu')(x) 
        x = BatchNormalization()(x)
        x = Dense(14,activation='relu')(x) 
        x = BatchNormalization()(x)
        x = Dense(18,activation='relu')(x) 
        x = BatchNormalization()(x)
        x = Dense(20,activation='relu')(x) 
        x = BatchNormalization()(x)                
      
        z_mean = Dense(latent_dim, name='z_mean',activation='sigmoid')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
                
        y_label= Input(shape=(latent_dim,)) 
        
        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
        
        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')       
        x= Dense(20,activation='relu')(latent_inputs)
        x = BatchNormalization()(x)
        x= Dense(18,activation='relu')(x)
        x = BatchNormalization()(x)
        x= Dense(14,activation='relu')(x)
        x = BatchNormalization()(x)
        x= Dense(10,activation='relu')(x)
        x = BatchNormalization()(x)
        x= Dense(6,activation='relu')(x)
        x = BatchNormalization()(x)
        outputs1 = Dense(original_dim, activation='sigmoid')(x)
                
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs1, name='decoder')
        decoder.summary()
        plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
        
        # instantiate VAE model
        
        outputs2 = [encoder(inputs)[0],decoder(encoder(inputs)[2])]
        vae = Model(inputs, outputs2, name='vae_mlp')
    
    # In[]
        # if __name__ == '__main__':
        '''parser = argparse.ArgumentParser()
        help_ = "Load h5 model trained weights"
        parser.add_argument("-w", "--weights", help=help_)
        help_ = "Use mse loss instead of binary cross entropy (default)"
        parser.add_argument("-m",
                            "--mse",
                            help=help_, action='store_true')
        args = parser.parse_args()
        models = (encoder, decoder)
        data = (Test_X, Test_Y)
        '''
        # VAE loss = mse_loss or xent_loss + kl_loss
        def mse1(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true), axis=-1) #+0.002*K.sum(y_pred*y_pred)
        
        def vae_loss(inputs,outputs):
            lossname='mse'
            #if args.mse:
            if lossname=='mse':
                reconstruction_loss = mse(inputs, outputs)
                
            else:                         
                reconstruction_loss = binary_crossentropy(inputs,outputs)
            
            reconstruction_loss *= original_dim
            kl_loss = 1 + z_log_var - K.square(z_mean-outputs2[0]) - K.exp(z_log_var)   # z_log_var=ln(sigma**2）
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5        
            y_loss=mse(z_mean,outputs2[0])
            return K.mean(reconstruction_loss + kl_loss) 
          
        vae.compile(optimizer=OptiFun,loss=[lossfun1,vae_loss],loss_weights=[0.8,0.2])  ## loss_weights=[0.05,0.95]
        
        vae.summary()
        plot_model(vae,
                   to_file='vae_mlp.png',
                   show_shapes=True)
    
        '''if args.weights:
            vae.load_weights(args.weights)
        else:
        '''
            # train the autoencoder
        def scheduler(epoch):    # 每隔100个epoch，学习率减小为原来的1/10
          if epoch % 5 == 0 and epoch != 0:
            lr = K.get_value(vae.optimizer.lr)
            K.set_value(vae.optimizer.lr, lr * 0.8)
            print("lr changed to {}".format(lr * 0.8))
          return K.get_value(vae.optimizer.lr)
      
        reduce_lr = LearningRateScheduler(scheduler)
        
    # In[]        
        hst=vae.fit(Train_X, [Train_Y,Train_X],                   
                    epochs=epochs,#callbacks=[reduce_lr],
                    batch_size=batch_size,shuffle=True,verbose=2,
                    validation_split=0.05)
    # In[]
        vae.save_weights('cvae_'+RegionAll[reg]+'.h5')
    # In[]    
    
        
        
    # A plot of loss on the training and validation datasets over training epochs
    # summarize history for loss
        plt.plot(hst.history['loss'])
        plt.plot(hst.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('2D_'+'eval_epoch'+str(epochs)+'.png') 
        plt.show()
            
        #X_dec=autoencoder.predict(Train_X)    
    #history.loss_plot('epoch') 
    
        y_pred_train = encoder.predict(Train_X)[0]  # NN预测的均值
        y_pred_test = encoder.predict(Test_X)[0]
        y_pred = encoder.predict(Predict_X)[0]  
        
        z_mean, z_sigma, z_z=encoder.predict(Test_X)    
        
        
        sigma_predAll=encoder.predict(Predict_X)[1] #NN预测的 ln(sigma**2)
        
        
       
        X_auto=decoder.predict(y_pred_test)
  
    
        MSE_test=np.sum((y_pred_test- Test_Y)**2)/Test_Y.shape[0]
        MSE_train=np.sum((y_pred_train- Train_Y)**2)/Train_Y.shape[0]
        
        
        ecludSim=np.arange(0.0,X_auto.shape[1])
        print('Region=',reg)
        
   
        
    #    Pearson coefficient
        for j in np.arange(0,X_auto.shape[1]):    
            ecludSim[j] = pearsonr(X_auto[:,j],Test_X[:,j])[0]
            print('j=',j,'  period=',period[j],' Pear =',ecludSim[j])   # X_auto与 Train_X之间的欧几里得相似度
        period=period[0:X_auto.shape[1]]    
        period=period.reshape(X_auto.shape[1],1)
        ecludSim=ecludSim.reshape(X_auto.shape[1],1)
        ressm= np.concatenate((period,ecludSim),axis=1) 
        
        np.savetxt('Res\\Similarity_'+region+'.txt',ressm,fmt="%8.2f %8.2f")
        print('Test MSE is:',MSE_test,'     Train MSE is:',MSE_train)
   
        x_pred_test=decoder.predict(y_pred_test)                         
     
        xstrname=eval('ss_x'+str(dispNo))
        ParaAll=['th','vp','vs','ro']   
        for NoPara in np.arange(0,4): 
            print('NoPara=',NoPara)
            para=ParaAll[NoPara] 
           
            strname=eval('ss_y'+str(NoPara+1))
            Y_earth=strname.inverse_transform(y_pred[:,NoPara:NoPara+1])   
          
            Y_earth=Y_earth[:,0:1]

            sigma_pred_tmp=sigma_predAll[:,NoPara:NoPara+1]
            sigma_pred=np.exp(0.5 * sigma_pred_tmp)   
            
            Y_earth_test= strname.inverse_transform(y_pred_test[:,NoPara:NoPara+1])  
            Y_earth_test=Y_earth_test[:,0:1]  
            Y_earth_test_X= xstrname.inverse_transform(Test_X[:,:])
            Y_earth_test_Y= strname.inverse_transform(Test_Y[:,NoPara:NoPara+1]) 
            Y_earth_test_Y=Y_earth_test_Y[:,0:1]
       
                   
            Y_earth_train= strname.inverse_transform(y_pred_train[:,NoPara:NoPara+1])  
            Y_earth_train=Y_earth_train[:,0:1] 
       
            Y_earth_train_X= xstrname.inverse_transform(Train_X[:,:])
            Y_earth_train_Y= strname.inverse_transform(Train_Y[:,NoPara:NoPara+1]) 
            Y_earth_train_Y=Y_earth_train_Y[:,0:1]
           
            
            [yy_Test,xx_Test]=vae.predict(Test_X)  
          
           
            res7=xstrname.inverse_transform(Test_X)
            res8=xstrname.inverse_transform(xx_Test)
           
           
            
            tmp1=np.loadtxt('China_2015_disp_v1.0\\SameRange_Longmen.dat')
           
            res1= np.concatenate((tmp1[:,0:1],tmp1[:,1:2],Y_earth),axis=1)  
            np.savetxt('Res\\'+region+'_'+para+'.txt',res1,fmt="%8.2f %8.2f %8.4f")      
           
                    
            res2=np.concatenate((tmp1[:,0:1],tmp1[:,1:2],sigma_pred),axis=1)  
            np.savetxt('Res\\'+region+'_'+para+'_sigma.txt',res2,fmt="%8.2f %8.2f %8.4f")     
           
   
            res3=np.concatenate((Y_earth_test,Y_earth_test_Y,y_pred_test[:,0:1],Test_Y[:,0:1],Y_earth_test-Y_earth_test_Y),axis=1)
              
        
    #  Sedimentary layer and crust thickness,avarage velocity and  density
else: 
    if(reg==6 or reg==7):
        for NoPara in np.arange(0,4):
            print('NoPara=',NoPara)
            para=ParaAll[NoPara]
            regstr=RegionAll[reg]
            tempu = np.loadtxt('Res\\'+regstr+'u_th.txt')
            tempm = np.loadtxt('Res\\'+regstr+'m_th.txt')
            templ = np.loadtxt('Res\\'+regstr+'l_th.txt')
            
            uh=tempu[:,2:3]
            mh=tempm[:,2:3]
            lh=templ[:,2:3]
             
            Resu = np.loadtxt('Res\\'+regstr+'u_'+para+'.txt')
            Resm = np.loadtxt('Res\\'+regstr+'m_'+para+'.txt')
            Resl = np.loadtxt('Res\\'+regstr+'l_'+para+'.txt')
           
            if(para=='th'):
                Res=np.concatenate((Resu[:,0:1],Resu[:,1:2],Resu[:,2:3]+Resm[:,2:3]+Resl[:,2:3]),axis=1)
 
            elif(para=='vp' or para=='vs'):
                Resu[:,2:3]=np.maximum(Resu[:,2:3], 0.00001)               
                Resm[:,2:3]=np.maximum(Resm[:,2:3], 0.00001) 
                Resl[:,2:3]=np.maximum(Resl[:,2:3], 0.00001)
               
                tempv=(uh+mh+lh)/(uh/Resu[:,2:3]+mh/Resm[:,2:3]+lh/Resl[:,2:3])
                #tempv=(uh*Resu[:,2:3]+mh*Resm[:,2:3]+lh*Resl[:,2:3])/(uh+mh+lh)
                Res=np.concatenate((Resu[:,0:1],Resu[:,1:2],tempv),axis=1)
                
            else:
                tempro=(uh*Resu[:,2:3]+mh*Resm[:,2:3]+lh*Resl[:,2:3])/(uh+mh+lh)
                Res=np.concatenate((Resu[:,0:1],Resu[:,1:2],tempro),axis=1)
            np.savetxt('Res\\'+regstr+'_'+para+'.txt',Res,fmt="%8.2f %8.2f %8.4f")

