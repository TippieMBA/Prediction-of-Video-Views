# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:15:01 2021

@author: RK
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def data_read(path):
    meta_data0 = pd.read_csv(path + '/Metadata/train_meta_df.csv')
    desc_data0 = pd.read_csv(path + '/Description Data/train_desc_df.csv')
    title_data0 = pd.read_csv(path + '/Title Data/train_title_df.csv')
    image_data0 = pd.read_csv(path + '/Image Data/train_image_df.csv')
    
    return meta_data0, desc_data0, title_data0, image_data0


def find_anomalies(data):
    #define a list to accumlate anomalies
    anomalies = []
    indx = []
    
    # Set upper and lower limit to 3 standard deviation
    random_data_std = data.std()
    random_data_mean = data.mean()
    anomaly_cut_off = random_data_std * 5
    
    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    for idx,outlier in data.items():
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
            indx.append(idx)
    return (anomalies,indx)

def data_scale(image_data):
    for col in image_data:
        means[col] = image_data[col].mean()
        maxs[col] = image_data[col].max()
        mins[col] = image_data[col].min()
    
    image_data = (image_data - image_data.mean()) / (image_data.max() - image_data.min())
    return image_data

def visuals(map_type, data, num_col = 1):
    if num_col == 1:
        print("data histogram")
        hist, bins, _ = plt.hist(meta_data['views'].values, bins=50)
        print("data's log histogram")
        hist, bins, _ = plt.hist(log(meta_data['views'].values), bins=50)
    
    else:
        a = np.random.random((16, 16))
        plt.imshow(image_data, cmap='hot', interpolation='nearest')
        plt.show()
        

def data_loader(image_features,meta_data,desc_data,title_features):
    meta_features = meta_data.drop(['comp_id','views','embed'],axis = 1)
    desc_features = desc_data.drop(['comp_id'],axis = 1)
    title_features = title_data.drop(['comp_id'],axis = 1)
    
    X_train_meta, X_test_meta, y_train_meta, y_test_meta = train_test_split(image_meta, labels_raw, test_size=0.2, random_state=6250)
    X_train_meta, X_valid_meta, y_train_meta, y_valid_meta = train_test_split(X_train_meta, y_train_meta, test_size=0.2, random_state=6250)

    torch.manual_seed(6250)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(6250)

    trainset = TensorDataset(torch.from_numpy(X_train_meta.astype('float32')).unsqueeze(1), torch.from_numpy(y_train_meta.astype('float32')))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)

    validset = TensorDataset(torch.from_numpy(X_valid_meta.astype('float32')).unsqueeze(1), torch.from_numpy(y_valid_meta.astype('float32')))
    validloader = torch.utils.data.DataLoader(validset, batch_size=10, shuffle=False, num_workers=2)

    testset = TensorDataset(torch.from_numpy(X_test_meta.astype('float32')).unsqueeze(1), torch.from_numpy(y_test_meta.astype('float32')))
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)
    
    return trainloader, validloader,testloader  
            
    