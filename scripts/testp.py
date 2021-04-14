# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 23:22:27 2021

@author: RK

"""

#%%
import Preprocess as prep
import numpy as np
import Train as train
import Model_eval as model_eval
from matplotlib import pyplot as plt
#%matplotlib inline
#%matplotlib inline
from numpy import log
#%%
path = 'data'
meta_data0,desc_data0,title_data0,image_data0 = prep.data_read(path)
meta_data1=meta_data0
desc_data1=desc_data0
title_data1=title_data0
image_data1=image_data0



#%%
prep.visuals('hist',meta_data1['views'].values)

#%%
anomay_index = prep.find_anomalies(meta_data1.views)
meta_data=meta_data1.drop(anomay_index[1]).reset_index(drop=True)
desc_data=desc_data1.drop(anomay_index[1]).reset_index(drop=True)
title_data=title_data1.drop(anomay_index[1]).reset_index(drop=True)
image_data=image_data1.drop(anomay_index[1]).reset_index(drop=True)

#%%
meta_data['ad_blocked']=meta_data.ad_blocked.astype(int)
meta_data['embed']=meta_data.embed.astype(int)
meta_data['partner']=meta_data.ad_blocked.astype(int)
meta_data['partner_active']=meta_data.ad_blocked.astype(int)

image_data = image_data.iloc[:,1:4001] # change dimension for full data
meta_data = meta_data
desc_data = desc_data
title_data = title_data




#%%
image_data = prep.data_scale(image_data)
desc_data = prep.data_scale(desc_data)
title_data = prep.data_scale(title_data)


#%%
#print(image_data)
prep.visuals('heat',image_data.values)
a = np.random.random((16, 16))
plt.imshow(image_data.values, cmap='hot', interpolation='nearest')
plt.show()

#%%
image_features = np.array(image_data)
labels_raw= log(meta_data['views'].values.reshape(-1,1).astype(float))
meta_features = meta_data.drop(['comp_id','views','embed'],axis = 1)
desc_features = desc_data.drop(['comp_id'],axis = 1)
title_features = title_data.drop(['comp_id'],axis = 1)


#%%
pca_components=2
data, explain_variance = prep.dim_reduction(meta_features, pca_components)
#print(np.sum(explain_variance))
#print(data)
#%%

image_meta = np.concatenate((image_features,meta_features.values,desc_features.values,title_features.values),axis=1)

#print(image_meta.shape)
#print(labels_raw.shape)
#%%
trainloader, validloader, testloader = prep.data_loader(image_meta,labels_raw)

#%%
dataiter = iter(trainloader)
X_samples, y_samples = dataiter.next()



#%%
l_rate = 0.0001
model = 'full'
train_error, valid_Error = train.train_model(trainloader,validloader, model, learning_rate=l_rate)
np.savetxt("error0.csv", 
           [train_error,valid_Error],
           delimiter =", ", 
           fmt ='% f')

#%%
model_eval.eval_plot(train_error,valid_Error)
