# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 14:44:13 2023

@author: TZU-HSUAN HUANG
"""

import torch
print(torch.__version__)
print(torch.cuda.is_available())
import os
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models.detection as detection
torch.cuda.get_device_name(0)
import copy

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import csv
from pathlib import Path
from torch.utils.data import random_split
from tqdm import tqdm

###########################################################################
##                          Argparse Setting                             ##
###########################################################################
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", default='D:/Master/2023ML/Lab2/Data/train_agu_2', help="1st-level folder of train data")
    parser.add_argument("--train_overview", default='D:/Master/2023ML/Lab2/train_data_agu_2.csv', help="path of overview file for training data")
    parser.add_argument("--test_data_path", default='D:/Master/2023ML/Lab2/Data/test-final', help="folder of testing data")

    parser.add_argument("--n_epochs_1", type=int, default=15, help="number of epochs of first-stage training")
    parser.add_argument("--n_epochs_2", type=int, default=30, help="number of epochs of second-stage training")
    parser.add_argument("--num_class", type=int, default=50, help="number of categories for classification")

    parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers for Dataloader")
    parser.add_argument("--lr", type=float, default=0.0035, help="adam: learning rate")
    parser.add_argument("--opt",  default='Adam', help="Optimizer:'SGD','RMSprop','Adagrad','Adam'")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.97, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--loss", default='Cross_entropy', help="Loss func:'MAE','MSE','NLL(Negative Log-Likelihood)','Cross_entropy'")

    parser.add_argument('--outf', default='D:/Master/2023ML/Lab2/Result', help='folder to output record and model')
    parser.add_argument('--log', default='D:/Master/2023ML/Lab2/Resultrecord_2.txt', help='path to record')
    parser.add_argument('--font', default=20, help='font size for output figure')
    parser.add_argument('--resume', default='', help='path to resume model weight')
    args = parser.parse_args(args=[])
    return args


###########################################################################
##                            Categories                                ##
###########################################################################
Categories = {0  : "abraham_grampa_simpson",
              1  : "agnes_skinner",
              2  : "apu_nahasapeemapetilon",
              3  : "barney_gumble",
              4  : "bart_simpson",
              5  : "brandine_spuckler",
              6  : "carl_carlson",
              7  : "charles_montgomery_burns",
              8  : "chief_wiggum",
              9  : "cletus_spuckler",
              10 : "comic_book_guy",
              11 : "disco_stu",
              12 : "dolph_starbeam",
              13 : "duff_man",
              14 : "edna_krabappel",
              15 : "fat_tony",
              16 : "gary_chalmers",
              17 : "gil",
              18 : "groundskeeper_willie",
              19 : "homer_simpson",
              20 : "jimbo_jones",
              21 : "kearney_zzyzwicz",
              22 : "kent_brockman",
              23 : "krusty_the_clown",
              24 : "lenny_leonard",
              25 : "lionel_hutz",
              26 : "lisa_simpson",
              27 : "lunchlady_doris",
              28 : "maggie_simpson",
              29 : "marge_simpson",
              30 : "martin_prince",
              31 : "mayor_quimby",
              32 : "milhouse_van_houten",
              33 : "miss_hoover",
              34 : "moe_szyslak",
              35 : "ned_flanders",
              36 : "nelson_muntz",
              37 : "otto_mann",
              38 : "patty_bouvier",
              39 : "principal_skinner",
              40 : "professor_john_frink",
              41 : "rainier_wolfcastle",
              42 : "ralph_wiggum",
              43 : "selma_bouvier",
              44 : "sideshow_bob",
              45 : "sideshow_mel",
              46 : "snake_jailbird",
              47 : "timothy_lovejoy",
              48 : "troy_mcclure",
              49 : "waylon_smithers"}


###########################################################################
##                              Preparing Data                          ##
###########################################################################

#---------------------------- Dataloader ---------------------------#
class Simpsons_DataSet(Dataset):
    def __init__(self, mode):

        args = parse_config()

        self.img_path = args.train_data_path
        self.img_names = np.squeeze(pd.read_csv(args.train_overview, usecols=["name"]).values)
        self.labels    = np.squeeze(pd.read_csv(args.train_overview, usecols=["label"]).values)
        assert len(self.img_names)==len(self.labels), 'Error : Data and labels length not the same'
        self.data_len=len(self.img_names)

        # Define the image augmentation transformations
        self.transformations = T.Compose([
                #T.ToPILImage(),  # Convert tensor back to PIL image for saving
                #T.AugMix(severity= 6,mixture_width=2),
                #T.RandomPosterize(bits=2, p=0.1),
                T.ToTensor(),

                #T.RandomApply([T.RandomHorizontalFlip()], p=0.1),
                #T.RandomApply([T.RandomVerticalFlip()], p=0.1),
                #T.RandomApply([T.RandomRotation(10)], p=0.1),

                #T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
                #T.RandomGrayscale(p=0.1),
                #T.RandomInvert(p=0.1),

                #T.RandomApply([T.RandomSolarize(threshold=1.0)], p=0.05),
                #T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),

                #T.RandomApply([AddGaussianNoise(0., 0.05)], p=0.1),  # mean and std
                #T.RandomApply([AddPoissonNoise(lam=0.1)], p=0.1),  # mean and std
                #T.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.1),
                #T.RandomApply([AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.1),

                #T.RandomApply([T.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
                #T.RandomApply([T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
                #T.RandomApply([T.ElasticTransform(alpha=250.0)], p=0.1),

                #T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),

                #T.RandomApply([AddGaussianNoise(0., 0.001)], p=1.0),  # mean and std

                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

            ])


        print(f'>> Found {self.data_len} images for training ...')
        with open(args.log, 'a') as f:
            f.write(f'>> Found {self.data_len} images for training ...\n')
            f.write(f'\n')
    def __len__(self):
        return self.data_len

    def __getitem__(self, index):


        single_img_name=os.path.join(self.img_path,self.img_names[index]+'.jpg')

        single_img=Image.open(single_img_name).convert("RGB")
        single_img = T.Resize((224, 224))(single_img)
        img=self.transformations(single_img)
        label=self.labels[index]
        #print(f'labels : {label}')

        return img, label

###########################################################################
##                                   Model                              ##
###########################################################################

class ResNet50(nn.Module):
    def __init__(self,num_class,pretrained=False):
        """
        Args:
            num_class: #target class
            pretrained:
                True: the model will have pretrained weights, and only the last layer's 'requires_grad' is True(trainable)
                False: random initialize weights, and all layer's 'require_grad' is True
        """
        super(ResNet50,self).__init__()
        self.model=models.efficientnet_v2_s(pretrained=pretrained)
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad=False

        num_neurons = self.model.classifier[1].in_features
        #num_neurons=self.model.fc.in_features
        #num_neurons = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_neurons, num_class, bias=True)
        #self.model.fc=nn.Linear(num_neurons, num_class, bias=True)


    def forward(self,X):
        out=self.model(X)
        return out

###########################################################################
##                         Train & Validation                           ##
###########################################################################
def train(model, train_loader, valid_loader, epochs, loss_func, optimizer, device, state):
    """
    Args:
        model: resnet model
        train_loader: training dataloader
        valid_loader: validation dataloader
        loss_func: loss function
        optimizer: optimizer
        epochs: number of training epoch
        device: gpu/cpu
        state: the state of the last training (used for resuming)
    Returns:
        df_loss: with column 'epoch','loss_train','loss_valid'
        df_acc: with column 'epoch','acc_train','acc_valid'
        best_model_wts: the best accuracy saved model
    """
    args = parse_config()
    num_class = args.num_class
    df_loss=pd.DataFrame()
    df_acc=pd.DataFrame()
    df_loss['epoch']=range(1,epochs+1)
    df_acc['epoch']=range(1,epochs+1)
    best_model_wts=None

    model.to(device)

    if args.resume != '':

        checkpoint = torch.load(os.path.join(args.resume, 'Model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        #saved_param = torch.load(os.path.join(args.resume, 'param.pth'))
        best_evaluated_acc = checkpoint['best_acc']
        start_epoch =checkpoint['last_epoch']+1
        loss_train=checkpoint['loss_train']
        loss_valid=checkpoint['loss_valid']
        acc_train=checkpoint['acc_train']
        acc_valid=checkpoint['acc_valid']


    else:
      start_epoch = 1
      best_evaluated_acc = 0
      loss_train=list()
      loss_valid=list()
      acc_train=list()
      acc_valid=list()


    with open(args.log, 'a') as f:
        f.write(f'>>> Start training... \n')

    for epoch in range(start_epoch,epochs+1):
        """
        train
        """
        with torch.set_grad_enabled(True):
            model.train()
            t_loss=0
            correct=0
            for i, (img, label) in enumerate(tqdm(train_loader, desc='Epoch '+str(epoch))):

                img, label=img.to(device),label.to(device,dtype=torch.long)
                predict=model(img)
                loss=loss_func(predict,label)
                t_loss+=loss.item()
                correct+=predict.max(dim=1)[1].eq(label).sum().item()
                """
                update
                """
                optimizer.zero_grad()
                loss.backward()  # bp
                optimizer.step()
            t_loss/=len(train_loader.dataset)
            t_acc=100.*correct/len(train_loader.dataset)
            loss_train.append(t_loss)
            acc_train.append(t_acc)

        """
        evaluate
        """
        _, v_loss, v_acc=evaluate(model, valid_loader, loss_func, device)
        loss_valid.append(v_loss)
        acc_valid.append(v_acc)
        print(f'epoch{epoch:>2d} | Train loss:{t_loss:.4f} | Train acc:{t_acc:.2f} | Valid loss:{v_loss:.4f} | Valid acc:{v_acc:.2f}\n')
        with open(args.log, 'a') as f:
            f.write(f'epoch{epoch:>2d} | Train loss:{t_loss:.4f} | Train acc:{t_acc:.2f} | Valid loss:{v_loss:.4f} | Valid acc:{v_acc:.2f}\n')

        # update best_model_wts
        if v_acc > best_evaluated_acc:
            best_evaluated_acc=v_acc
            best_model_wts=copy.deepcopy(model)
            torch.save(model, os.path.join(args.outf, 'Best_Model.pth'))

        checkpoint_dict = {'last_epoch': epoch,
                           'model_state_dict': model.state_dict(),
                           'optim_state_dict': optimizer.state_dict(),
                           'best_acc' : best_evaluated_acc,
                           'loss_train' : loss_train,
                           'loss_valid' : loss_valid,
                           'acc_train' : acc_train,
                           'acc_valid' : acc_valid,
                           'state' : state}
        torch.save( checkpoint_dict, os.path.join(args.outf, 'Model.pth'))


    df_loss['loss_train']=loss_train
    df_loss['loss_valid']=loss_valid
    df_acc['acc_train']=acc_train
    df_acc['acc_valid']=acc_valid

    # save model
    #torch.save(best_model_wts,os.path.join(args.save_model+'.pt'))
    #model.load_state_dict(best_model_wts)

    return df_loss, df_acc, best_model_wts

def evaluate(model, valid_loader, loss_func, device):
    """
    Args:
        model: resnet model
        valid_loader: validation dataloader
        device: gpu/cpu
    Returns:
        confusion_matrix: (num_class,num_class) ndarray
        total_loss : validation average loss
        acc: validation accuracy
    """

    args = parse_config()
    num_class = args.num_class
    confusion_matrix=np.zeros((num_class,num_class))

    with torch.set_grad_enabled(False):
        model.eval()
        correct=0
        total_loss=0
        for img, label in valid_loader:
            img,label=img.to(device),label.to(device,dtype=torch.long)
            predict=model(img)
            predict_class=predict.max(dim=1)[1]
            loss=loss_func(predict,label)
            total_loss+=loss.item()
            correct+=predict_class.eq(label).sum().item()
            for i in range(len(label)):
                confusion_matrix[int(label[i])][int(predict_class[i])]+=1
        total_loss/=len(valid_loader.dataset)
        acc=100.*correct/len(valid_loader.dataset)

    # normalize confusion_matrix
    confusion_matrix=confusion_matrix/confusion_matrix.sum(axis=1).reshape(num_class,1)

    return confusion_matrix, total_loss,acc

###########################################################################
##                               Plot                                   ##
###########################################################################
def plot(dataframe1, mode):
    """
    Arguments:
        dataframe1: dataframe with 'epoch','loss_train','loss_valid','acc_train','acc_valid' columns of without pretrained weights model
        title: figure's title
    Returns:
        figure: an figure
    """
    fig=plt.figure(figsize=(10,6))
    for name in dataframe1.columns[1:]:
        plt.plot(range(1,1+len(dataframe1)),name,data=dataframe1,label=name)

    plt.xlabel('Epochs')
    if mode == 'loss' :
        plt.ylabel('Loss')
        plt.title('Learning curve (loss)')
    elif mode == 'acc':
        plt.ylabel('Accuracy(%)')
        plt.title('Learning curve (acc)')
    plt.legend()
    return fig

def plot_confusion_matrix(confusion_matrix):
    args = parse_config()
    label_name = ["abraham_grampa_simpson", "agnes_skinner", "apu_nahasapeemapetilon", "barney_gumble", "bart_simpson",
                  "brandine_spuckler", "carl_carlson", "charles_montgomery_burns", "chief_wiggum", "cletus_spuckler",
                  "comic_book_guy", "disco_stu", "dolph_starbeam", "duff_man", "edna_krabappel",
                  "fat_tony", "gary_chalmers", "gil", "groundskeeper_willie", "homer_simpson",
                  "jimbo_jones", "kearney_zzyzwicz", "kent_brockman", "krusty_the_clown", "lenny_leonard",
                  "lionel_hutz", "lisa_simpson", "lunchlady_doris", "maggie_simpson", "marge_simpson",
                  "martin_prince", "mayor_quimby", "milhouse_van_houten", "miss_hoover", "moe_szyslak",
                  "ned_flanders", "nelson_muntz", "otto_mann", "patty_bouvier", "principal_skinner",
                  "professor_john_frink", "rainier_wolfcastle", "ralph_wiggum", "selma_bouvier", "sideshow_bob",
                  "sideshow_mel", "snake_jailbird", "timothy_lovejoy", "troy_mcclure", "waylon_smithers"]

    fig, ax = plt.subplots(figsize=(25,25))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    ax.xaxis.set_label_position('top')

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(i, j, '{:.2f}'.format(confusion_matrix[j, i]), va='center', ha='center')
    ax.set(xticks=range(0, 50, 1), xticklabels=label_name, yticks=range(0, 50,1), yticklabels=label_name)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    ax.set_xlabel('Predicted label', fontsize = 15)
    ax.set_ylabel('True label', fontsize = 15)
    return fig

###########################################################################
##                                   Test                               ##
###########################################################################
def test(model, outf):
    """
    Args:
        model: trained model
        testing_dataset: testing dataset
        outf: path to output file
    """
    args = parse_config()
    df_id =list()
    df_character = list()
    img_path = args.test_data_path
    # Define the image augmentation transformations
    transformations = T.Compose([
        T.ToTensor(),  # Convert tensor back to PIL image for saving
        T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    img_names =list()
    count = 0
    with torch.set_grad_enabled(False):
        model.eval()
        correct=0
        i = 1
        for filename in os.listdir(img_path):
            count=count+1
            file_name=Path(filename).stem
            img_names.append(file_name)
            single_img_name=os.path.join(img_path,file_name+'.jpg')
            single_img=Image.open(single_img_name).convert("RGB")
            single_img = T.Resize((224, 224))(single_img)
            img=transformations(single_img)
            img=img.to(device)
            predict=model(img.unsqueeze(0))
            df_id.append(file_name)
            print(torch.argmax(predict,1))
            df_character.append(Categories.get(int(torch.argmax(predict,1))))
            i=i+1

    data_len = count
    df = pd.DataFrame()

    df['id'] = df_id
    df['character'] = df_character
    df.to_csv(os.path.join(outf, 'test.csv'))

    return data_len, df

###########################################################################
##                               Main                                   ##
###########################################################################

if __name__ == '__main__':

    args = parse_config()
    if args.resume == '':
      if os.path.exists(args.log):
        os.remove(format(args.log))


      with open(args.log, 'a') as f:
          f.write(f'----------------------------------------------------------\n')
          f.write(f'-                    Hyperparameters                     -\n')
          f.write(f'----------------------------------------------------------\n')
          f.write(f'>> Epoch :                  first stage :{args.n_epochs_1} | second stage :{args.n_epochs_2}\n')
          f.write(f'>> Optimizer :              {args.opt}\n')
          f.write(f'>> Learning rate :          {args.lr}\n')
          f.write(f'>> Loss func :              {args.loss}\n')
          f.write(f'>> Batch size :             {args.batch_size}\n')


      with open(args.log, 'a') as f:
          f.write(f'----------------------------------------------------------\n')
          f.write(f'-                    Training......                      -\n')
          f.write(f'----------------------------------------------------------\n')

    """
        Data Loading
    """
    #---- Train & Valid data loading ----#
    train_dataset = Simpsons_DataSet(mode='train')
    ## Train & valid split
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    ## Train & valid dataloader
    train_loader=DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader=DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)


    """
        Train
    """
    print(">>> Start training...\n")
    ## Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## Loss Func
    if args.loss == 'MAE' :
        loss_func = nn.L1Loss()
    elif args.loss == 'MSE' :
        loss_func = nn.MSELoss()
    elif args.loss == 'NLL' :
        loss_func = nn.NLLLoss()
    elif args.loss == 'Cross_entropy' :
        loss_func = nn.CrossEntropyLoss()
    ## Train calling
    """
    resnet50 with pretrained weights
        first feature extraction for few epochs, then finefuning for some epochs
    """

    model_with=ResNet50(num_class=args.num_class, pretrained=True)

    if args.resume != '':
        checkpoint = torch.load(os.path.join(args.resume, 'Model.pth'))
        best_evaluated_acc = checkpoint['best_acc']
        state =checkpoint['state']
    else :
        state = 0


    if state == 0 :

        # feature extraction
        print(f'>>> Feature extraction...\n')
        with open(args.log, 'a') as f:
            f.write(f'>>> Feature extraction...\n')

        params_to_update=[]
        for name,param in model_with.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        optimizer=optim.Adam(params_to_update, lr=args.lr, betas=(args.b1, args.b2), eps=1e-08, weight_decay=2e-5, amsgrad=True)
        df_loss_1, df_acc_1, best_model_wts_1=train(model_with, train_loader, valid_loader, args.n_epochs_1, loss_func,optimizer, device, state)
        state = 1
        torch.save({'df_loss_1' : df_loss_1,
                    'df_acc_1' : df_acc_1},
                    '%s/loss_acc.pth' % args.outf)

        # finetuning
        print(f'>>> Fine-tuning...\n')
        with open(args.log, 'a') as f:
            f.write(f'>>> Fine-tuning...\n')
        for param in model_with.parameters():
            param.requires_grad=True
        optimizer=optim.Adam(model_with.parameters(), lr=args.lr, betas=(args.b1, args.b2), eps=1e-08, weight_decay=2e-5, amsgrad=True)
        df_loss_2, df_acc_2, best_model_wts_2=train(model_with, train_loader, valid_loader, args.n_epochs_2, loss_func,optimizer, device, state)
        state = 2

    elif state == 1 :
        # finetuning
        saved_loss_acc = torch.load(os.path.join(args.resume, 'loss_acc.pth'))
        df_loss_1 = saved_loss_acc['df_loss_1']
        df_acc_1 = saved_loss_acc['df_acc_1']
        print(f'>>> Fine-tuning...\n')
        #with open(args.log, 'a') as f:
            #f.write(f'>>> Fine-tuning...\n')
        for param in model_with.parameters():
            param.requires_grad=True
        optimizer=optim.Adam(model_with.parameters(), lr=args.lr, betas=(args.b1, args.b2), eps=1e-08, weight_decay=2e-5, amsgrad=True)
        df_loss_2, df_acc_2, best_model_wts_2=train(model_with, train_loader, valid_loader, args.n_epochs_2, loss_func,optimizer, device, state)

        state = 2


    df_loss = pd.concat([df_loss_1,df_loss_2],axis=0,ignore_index=True)
    df_acc = pd.concat([df_acc_1,df_acc_2],axis=0,ignore_index=True)

    """
        plot & save
    """

    #---- Plot confusion matrix ----#
    confusion_matrix, _, __ = evaluate (torch.load(os.path.join(args.outf, 'Best_Model.pth')), valid_loader, loss_func, device)
    figure=plot_confusion_matrix(confusion_matrix)
    figure.savefig(os.path.join(args.outf, 'Confusion matrix.png'))

    #---- Plot learning curve ----#
    figure=plot(df_loss,'loss')
    figure.savefig(os.path.join(args.outf, 'Learning curve (loss).png'))

    figure=plot(df_acc,'acc')
    figure.savefig(os.path.join(args.outf, 'Learning curve (acc).png'))

    #---- Output predictions of testing data ----#
    with open(args.log, 'a') as f:
        f.write(f'----------------------------------------------------------\n')
        f.write(f'-                     Testing......                      -\n')
        f.write(f'----------------------------------------------------------\n')
    test_len, prediction = test(torch.load(os.path.join(args.outf, 'Best_Model.pth')), args.outf)
    with open(args.log, 'a') as f:
        f.write(f'>> Found {test_len} images for testning ...\n')
        f.write(f'\n')
        f.write(f'>> Prediction :\n')
        f.write(f'{prediction}')
