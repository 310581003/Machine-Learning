# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 14:56:41 2023

@author: TZU-HSUAN HUANG
"""

import torch
print(torch.__version__)
print(torch.cuda.is_available())
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models.detection as detection
torch.cuda.get_device_name(0)

import copy
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import csv

###########################################################################
##                      Argparse Setting                                 ##
###########################################################################
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default='D:/Master/2023ML/Lab1/train_norm.csv', help="path of training data file")
    parser.add_argument("--valid_data", default='D:/Master/2023ML/Lab1/valid_norm.csv', help="path of validation data file")
    parser.add_argument("--test_data", default='D:/Master/2023ML/Lab1/test_norm.csv', help="path of testinging data file")
    
    parser.add_argument("--feature_sel", default='all', help="input feature selection")
    parser.add_argument("--n_epochs", type=int, default=800, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=5500, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.025, help="adam: learning rate")
    parser.add_argument("--opt",  default='Adam', help="Optimizer:'SGD','RMSprop','Adagrad','Adam'")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.97, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--loss", default='MAE', help="Loss func:'MAE','MSE','NLL(Negative Log-Likelihood)','Cross_entropy'")
    
    parser.add_argument('--outf', default='D:/Master/2023ML/Lab1/Result/', help='folder to output record and model')
    parser.add_argument('--log', default='D:/Master/2023ML/Lab1/Result/record.txt', help='path to record')
    parser.add_argument('--font', default=20, help='font size for output figure')
    parser.add_argument('--save_model', default='D:/Master/2023ML/Lab1/Result/Saved_Model', help='path to resume model weight')

    args = parser.parse_args(args=[])
    return args


###########################################################################
##                        Preparing Data                                 ##
###########################################################################
def Sketch_Data (mode, feature_sel) :
    
    args = parse_config()
    
    ## read data for different modes
    if mode == 'train':
        data = pd.read_csv(args.train_data, usecols=feature_sel).values        
        labels=pd.read_csv(args.train_data, usecols=["price"]).values
        
    elif mode == 'valid':
        data = pd.read_csv(args.valid_data, usecols=feature_sel).values
        labels=pd.read_csv(args.valid_data, usecols=["price"]).values
        
    elif mode == 'test':
        data = pd.read_csv(args.test_data, usecols=feature_sel).values
        labels={}
            
    print (f'Data:\n{data}')
    print (f'Label:\n{labels}')
    
    ## data length calculation
    data_len=len(data)
    print(f'>> Found {data_len} PPG segments for {mode}...')
    with open(args.log, 'a') as f:
        f.write(f'>> Found {data_len} data for {mode}...\n')
        f.write(f'\n')
    return data, labels

def collate_fn(data):
    img, bbox = data
    zipped = zip(img, bbox)
    return list(zipped)
###########################################################################
##                            Model                                      ##
###########################################################################

class Model(nn.Module):
    def __init__(self, num_input):
        super(Model, self).__init__()
        self.num_input = num_input        
        self.layer = nn.Sequential( 
                        nn.Linear(self.num_input , self.num_input, bias=True),
                        #nn.BatchNorm1d(self.num_input),
                        nn.ELU(),                      
                        nn.Linear(self.num_input, self.num_input*2, bias=True),
                        #nn.BatchNorm1d(self.num_input*3),
                        nn.ELU(),
                        #nn.Dropout(0.6),
                        nn.Linear(self.num_input*2, self.num_input*2, bias=True),
                        #nn.BatchNorm1d(self.num_input*3),
                        nn.ELU(),  
                        nn.Linear(self.num_input*2, self.num_input*2, bias=True),
                        #nn.BatchNorm1d(self.num_input*3),
                        nn.ELU(),
                        #nn.Dropout(0.5),
                        nn.Linear(self.num_input*2 , self.num_input*2, bias=True),
                        #nn.BatchNorm1d(self.num_input*2),
                        nn.ELU(),
                        #nn.Dropout(0.5),
                        nn.Linear(self.num_input*2 , self.num_input*2, bias=True),
                        #nn.BatchNorm1d(self.num_input*2),
                        nn.ELU(),
                        #nn.Dropout(0.5),
                        nn.Linear(self.num_input*2 , self.num_input, bias=True),
                        #nn.BatchNorm1d(self.num_input),
                        nn.ELU(),
                        nn.Linear(self.num_input , self.num_input, bias=True),
                        #nn.BatchNorm1d(self.num_input),
                        nn.ELU(),
                        nn.Linear(self.num_input , 1),
                        )
        '''
        self.layer = nn.Sequential(
                        nn.Linear(self.num_input, self.num_input*2, bias=True),
                        #nn.BatchNorm1d(42),
                        nn.ELU(),
                        #nn.Dropout(0.6),
                        nn.Linear(self.num_input*2, self.num_input*4, bias=True),
                        nn.ELU(),
                        #nn.BatchNorm1d(42),
                        nn.Linear(self.num_input*4 , self.num_input*4, bias=True),
                        #nn.BatchNorm1d(84),
                        nn.ELU(),
                        nn.Dropout(0.5),
                        nn.Linear(self.num_input*4 , self.num_input*4, bias=True),
                        #nn.BatchNorm1d(84),
                        nn.ELU(),
                        nn.Dropout(0.5),
                        nn.Linear(self.num_input*4 , self.num_input*4, bias=True),
                        #nn.BatchNorm1d(84),
                        nn.ELU(),
                        nn.Dropout(0.5),
                        nn.Linear(self.num_input*4 , self.num_input*2, bias=True),
                        #nn.BatchNorm1d(84),
                        nn.ELU(),
                        nn.Linear(self.num_input*2 , self.num_input, bias=True),
                        #nn.BatchNorm1d(84),
                        nn.ELU(),
                        nn.Linear(self.num_input , 1),
                        )
        '''
    def forward(self, x):
        x = self.layer(x)
        return x
    
    
###########################################################################
##                      Train & Validation                               ##
###########################################################################   

def train(model,loader_train,loader_valid,Loss,optimizer,epochs,device):
    """
    Args:
        model: resnet model
        loader_train: training dataloader
        loader_valid: validation dataloader
        Loss: loss function
        optimizer: optimizer
        epochs: number of training epoch
        device: gpu/cpu
    Returns:
        df_loss: with column 'epoch','loss_train','loss_test'
        best_model_wts : the trained model with the best loss 
    """
    args = parse_config()
    df_acc=pd.DataFrame()
    df_loss=pd.DataFrame()
    df_acc['epoch']=range(1,epochs+1)
    df_loss['epoch']=range(1,epochs+1)
    best_model_wts=None
    best_evaluated_loss=10000
    
    model.to(device)
    acc_train=list()
    acc_valid=list()
    loss_train=list()
    loss_valid=list()
    with open(args.log, 'a') as f:
        f.write(f'----------------------------------------------------------\n')
        f.write(f'-                        Training                        -\n')
        f.write(f'----------------------------------------------------------\n')
    for epoch in range(1,epochs+1):
        """
        train
        """
        with torch.set_grad_enabled(True):
            model.train()
            train_total_loss=0
            correct=0
            for data,labels in  (loader_train):
                data,labels=data.to(device),labels.to(device)
                predict=model(data.float())
                loss=Loss(predict,labels.float())
                train_total_loss+=loss.item()
                correct+=predict.max(dim=1)[1].eq(labels).sum().float().item()
                """
                update
                """
                optimizer.zero_grad()
                loss.backward()  # bp
                optimizer.step()
            train_total_loss/=len(loader_train.dataset)
            loss_train.append(train_total_loss)
            #acc=100.*correct/len(loader_train.dataset)
            #acc_train.append(acc)
            
            
        """
            validation
        """
        
        acc, valid_loss=evaluate(model,loader_valid,Loss,device)
        #acc_valid.append(acc)
        loss_valid.append(valid_loss)
        print(f'epoch{epoch:>2d} | Training loss:{train_total_loss:.4f} | Validation loss:{valid_loss:.4f}')
        with open(args.log, 'a') as f:
            f.write(f'epoch{epoch:>2d} | Training loss:{train_total_loss:.4f} | Validation loss:{valid_loss:.4f}\n')
        # update best_model_wts
        if valid_loss<best_evaluated_loss:
            best_evaluated_loss=valid_loss
            best_model_wts=copy.deepcopy(model)
        
    #for name, parms in model.named_parameters():
        #print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '-->weight:', torch.mean(parms.data), '-->grad_value:', torch.mean(parms.grad))
    #df_acc['acc_train']=acc_train
    #df_acc['acc_valid']=acc_valid
    df_loss['loss_train']=loss_train
    df_loss['loss_valid']=loss_valid 
    print(f'The best valid loss : {best_evaluated_loss}')
    with open(args.log, 'a') as f:
        f.write(f'>>>>>>> Best valid loss : {best_evaluated_loss}\n')
    # save model
    torch.save(best_model_wts,os.path.join(args.save_model+'.pt'))
    #model.load_state_dict(model)
    
    return df_loss, best_model_wts

def evaluate(model,loader_valid,Loss,device):
    """
    Args:
        model: resnet model
        loader_test: testing dataloader
        device: gpu/cpu
    Returns:
        confusion_matrix: (num_class,num_class) ndarray
        acc: validation accuracy
        total_loss : validation loss
    """
    
    with torch.set_grad_enabled(False):
        model.eval()
        correct=0
        total_loss=0
        for data,labels in  (loader_valid):  
            data,labels=data.to(device),labels.to(device)
            predict=model(data.float())
            loss=Loss(predict,labels.float())
            total_loss+=loss.item()
            predict_class=predict.max(dim=1)[1]
            correct+=predict_class.eq(labels).sum().float().item()
        total_loss/=len(loader_valid.dataset)
        
        acc=100.*correct/len(loader_valid.dataset) 
        
    return acc, total_loss
 
    
###########################################################################
##                            Plot                                       ##
###########################################################################
def plot(df_loss, title):
    """
    Arguments:
        df_loss: dataframe with 'epoch','loss_train','loss_test' columns of trained weights model  
        title: figure's title
    Returns:
        fig_loss : figure of the learning curve
    """
    fig_loss=plt.figure(figsize=(10,6))
    for name in df_loss.columns[1:]:
        plt.plot(range(1,1+len(df_loss)),name,data=df_loss,label=name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()

    return fig_loss

###########################################################################
##                            Test                                       ##
###########################################################################
def test(model,testing_dataset,outf):
    """
    Args:
        model: trained model
        testing_dataset: testing dataset
        outf: path to output file
    """
    df_id =list()
    df_price = list()
    with torch.set_grad_enabled(False):
        model.eval()
        correct=0
        i = 1
        for data in testing_dataset:  
            data=data.to(device)
            predict=model(data.float())
            df_id.append(i)
            df_price.append(predict.item())
            i=i+1 
    df = pd.DataFrame()
    df['id'] = df_id
    df['price'] = df_price
    df.to_csv(os.path.join(outf, 'test.csv')) 

 

###########################################################################
##                            Main                                       ##
###########################################################################
   
if __name__ == '__main__':
    
    args = parse_config()
    
    """
        Data Loading
    """
    ## Define input features
    all_feature = ['sale_yr', 'sale_month', 'sale_day', 'bedrooms', 'bathrooms', 'sqft_living',
                   'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                   'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
                   'sqft_lot15']
    '''
    all_feature = ['sale_yr', 'sale_month', 'sale_day', 'bedrooms', 'bathrooms', 'sqft_living',
                   'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                   'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
                   'sqft_lot15','cust_time_type', 'cust_layout_type', 'cust_area_type', 'cust_address_type', 'cust_condition_type', 'sale_long', 'more_than_1_floor', 'grade_more_than_8', 'basement_or_not',
                   'renovate_or_not', 'built_long']
    all_feature = ['sale_yr', 'sale_month', 'sale_day', 'bedrooms', 'bathrooms', 'sqft_living',
                   'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                   'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
                   'sqft_lot15']
    all_feature = ['bedrooms', 'bathrooms', 'sqft_living',
                   'sqft_lot', 'floors', 'waterfront', 'view', 'grade', 'sqft_above',
                   'sqft_basement', 'yr_renovated', 'lat', 'sqft_living15',
                   'cust_time_type', 'cust_layout_type', 'cust_condition_type', 'more_than_1_floor', 'grade_more_than_8', 'basement_or_not',
                   'renovate_or_not']
    all_feature = ['bedrooms', 'bathrooms', 'sqft_living',
                   'view', 'grade', 'sqft_above',
                   'sqft_basement', 'lat', 'sqft_living15',
                   'grade_more_than_8']
    '''
    if args.feature_sel == 'all':
        feature_sel = all_feature
    else :
        feature_sel= args.feature_sel
    num_input =  len(feature_sel)
    
    with open(args.log, 'a') as f:
        f.write(f'----------------------------------------------------------\n')
        f.write(f'-                  Data description                      -\n')
        f.write(f'----------------------------------------------------------\n')
    ## Dataloader
    training_data, training_target = Sketch_Data(mode='train', feature_sel= feature_sel)
    training_data = torch.from_numpy(training_data)
    training_target = torch.from_numpy(training_target)
    training_dataset = torch.utils.data.TensorDataset(training_data, training_target)
    loader_train = DataLoader(dataset=training_dataset, batch_size=args.batch_size,shuffle=True)
    
    valid_data, valid_target = Sketch_Data(mode='valid', feature_sel= feature_sel)
    valid_data = torch.from_numpy(valid_data)
    valid_target = torch.from_numpy(valid_target)
    valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_target)
    loader_valid = DataLoader(dataset=valid_dataset, batch_size=args.batch_size,shuffle=True)
    
    testing_data,   testing_target= Sketch_Data(mode='test', feature_sel= feature_sel)
    testing_data = torch.from_numpy(testing_data)
    #testing_target = torch.from_numpy(testing_target)
    #testing_dataset = torch.utils.data.TensorDataset(testing_data, testing_target)
    
    with open(args.log, 'a') as f:
        f.write(f'----------------------------------------------------------\n')
        f.write(f'-                    Hyperparameters                     -\n')
        f.write(f'----------------------------------------------------------\n')
        f.write(f'>> Epoch :                  {args.n_epochs}\n')
        f.write(f'>> Optimizer :              {args.opt}\n')
        f.write(f'>> Learning rate :          {args.lr}\n')
        f.write(f'>> Loss func :              {args.loss}\n')
        f.write(f'>> Batch size :             {args.batch_size}\n')
        f.write(f'>> # of input features :    {num_input}\n')
        f.write(f'>> Input features :         {feature_sel}\n')
    """
        Train
    """
    ## Device
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## Model
    model = Model(num_input)
    ## Optimizer
    if args.opt == 'SGD' :
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.opt == 'RMSprop' :
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    elif args.opt == 'Adagrad' :
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
    elif args.opt == 'Adam' :
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2), eps=1e-08, weight_decay=2e-5, amsgrad=True)
    ## Loss Func
    if args.loss == 'MAE' :
        loss_function = nn.L1Loss()
    elif args.loss == 'MSE' :
        loss_function = nn.MSELoss()
    elif args.loss == 'NLL' :
        loss_function = nn.NLLLoss()
    elif args.loss == 'Cross_entropy' :
        loss_function = nn.CrossEntropyLoss()
    ## Train calling
    df_loss, best_model_wts=train(model=model, loader_train=loader_train, loader_valid=loader_valid, 
                   Loss=loss_function, optimizer=optimizer, epochs=args.n_epochs, device=device)
    
    
    """
        plot accuracy figure
    """
    fig_loss = plot(df_loss,'Learning curve')
    fig_loss.savefig('Learning curve (Loss).png')
    fig_loss.savefig(args.outf +  'Learning_curve' +'.png')

    """
        Prediction for testing data
    """
    test(model=best_model_wts, testing_dataset=testing_data, outf=args.outf)
    