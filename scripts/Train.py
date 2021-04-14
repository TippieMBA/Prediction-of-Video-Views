# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:37:39 2021

@author: RK
"""

import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import Models as models
#from azureml.core import Run

# ADDITIONAL CODE: get AML run from the current context
#run = Run.get_context()

def train_model(trainloader,validloader, full_select, learning_rate=0.0001):
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
    if full_select == 'full':
        net = models.MyModel(meta_input = 11,meta_hidden=16, meta_output=12)
    elif full_select == 'res_net':
        net = models.ResNet(models.ResidualBlock)
    elif full_select == 'feed_forward':
        net = models.FeedForwardNet(n_input=11, n_hidden=16, n_output=1)
    else:
        print("choose correct parameters")
        return 0
    optimizer = optim.SGD(net.parameters(), lr=learning_rate) # for metadata
    criterion = nn.MSELoss()
    
    train_losses = []
    valid_losses = []
    num_epochs=50
    total_step=10

    for epoch in range(num_epochs):  # loop over the dataset multiple times

    # set the model as train mode
    #model.train()
        train_loss = 0.0
        train_counter = 0
        #running_loss_train = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, targets = data
        
            image_data1 = inputs[:,:,0:4000].unsqueeze(1)
            meta_data1 = inputs[:,:,4000:4011].squeeze(1)
            desc_featurs1 = inputs[:,:,4011:4061].unsqueeze(1)
            title_features1 = inputs[:,:,4061:4111].unsqueeze(1)
            #torch.autograd.set_detect_anomaly(True)
            #print(image_data1.shape)
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

        # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
            #outputs = net(meta_data1)
            if full_select == 'full':
                outputs = net(image_data1,meta_data1,desc_featurs1,title_features1)
            elif full_select == 'res_net':
                outputs = net(image_data1)
            elif full_select == 'feed_forward':
                outputs = net(meta_data1)
            else:
                print("choose correct parameters")
                return 0
            
        #print(outputs)
            loss = torch.sqrt(criterion(outputs, targets))
        #print(loss)
            loss.backward()
        #scheduler.step()
            optimizer.step()
            #print("training is going on   ")
            #print(i)
            #running_loss_train += loss.item()
            if i % 2000 == 1:
                #print ('Epoch [{}/{}], Step [{}], Loss: {:.4f}' 
                #       .format(epoch+1, num_epochs, i+1:5, loss.item()))

                print(f'epoch={epoch + 1}, batch={i + 1:5}: loss {loss:.2f}')
        
            train_loss += (loss.item() * inputs.size(0))
            train_counter += inputs.size(0)
        #print(train_loss)
        train_losses.append(train_loss/train_counter)
    
    # switch to evaluation mode
    #model.eval()
        valid_loss = 0.0
        valid_counter = 0
        with torch.no_grad():
            for i, data in enumerate(validloader, 0):
            # get the inputs
                inputs, targets = data
                inputs, targets = data
                image_data1 = inputs[:,:,0:4000].unsqueeze(1)
                meta_data1 = inputs[:,:,4000:4011].squeeze(1)
                desc_featurs1 = inputs[:,:,4011:4061].unsqueeze(1)
                title_features1 = inputs[:,:,4061:4111].unsqueeze(1)

                if cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                if full_select == 'full':
                    outputs = net(image_data1,meta_data1,desc_featurs1,title_features1)
                elif full_select == 'res_net':
                    outputs = net(image_data1)
                elif full_select == 'feed_forward':
                    outputs = net(meta_data1)
                else:
                    print("choose correct parameters")
                    return 0
          #outputs = modelX(image_data1,meta_data1,desc_featurs1,title_features1)
                loss = criterion(outputs, targets)

                valid_loss += (loss.item() * inputs.size(0))
                valid_counter += inputs.size(0)
    
            valid_losses.append(valid_loss/valid_counter)
    
    print('Finished Training')
    return train_losses,valid_losses
    
    