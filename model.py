from __future__ import print_function, division
import torch
import os
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
import sklearn.metrics as sklm
import csv
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os

import pickle
import random
from shutil import copyfile
from shutil import rmtree

import torchvision

use_gpu = torch.cuda.is_available()
print("We see GPU:")
print(use_gpu)
print("Let's use", torch.cuda.device_count(), "GPUs!")

from PIL import Image

import CXRDataset as CXR
import Eval as E
from importlib import reload
reload(CXR)
reload(E)

def checkpoint(model_ft, best_acc, best_loss, epoch,PRED_LABEL,LR,RESULT_PATH):
    """
    save checkpoint
    
    args:
        model_ft: torchvision model
        best_acc: best accuracy achieved so far in training
        best_loss: best loss achieved so far in training
        epoch: last epoch of training
        PRED_LABEL: what we're predicting; expect format ["Pneumonia"] or ["Pneumonia","Opacity"]... etc
        LR: learning rate
        RESULT_PATH: path to save this to
    returns:
        nothing (saves file)

    """
    # Save checkpoint.
    print('Saving..')
    state = {
        'model_ft': model_ft,
        'best_acc': best_acc,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR':LR         
    }

    torch.save(state, RESULT_PATH+'checkpoint_'+PRED_LABEL)
    

def train_model(model, criterion, optimizer, LR, num_epochs=5,dataloaders="x",dataset_sizes="x", PRED_LABEL="x", start_epoch=1,MULTILABEL=True,FOLD_OVERRIDE="",TRAIN_FILTER="",RESULT_PATH="results/",MULTICLASS=False):
    """
    performs torchvision model training
    
    args:
        model: model to fine tune
        criterion: pytorch optimization criteria
        optimizer: pytorch optimizer
        LR: learning rate
        num_epochs: stop after this many epochs
        dataloaders: torchvision dataloader
        dataset_sizes: length of train/val datasets 
        PRED_LABEL: targets we're predicting in list format ["PNA","Opacity"] etc
        start_epoch: in case of loading saved model; not currently used
        MULTILABEL: should be removed - always True - everything is trained using multilabel list format now even single labels ["Pneumonia"]
        FOLD_OVERRIDE: columns of scalars with train/val/test split
        TRAIN_FILTER: list of data we're training on, used for labeling results
        RESULT_PATH= path at which resutls are saved, recommend leaving default to use with other scripts
        MULTICLASS: if training on single multiclass n>2 target; currently only implemented for single multiclass target.     
    returns:
        model: trained torchvision model
        best_epoch: epoch on which best model was achieved
    
    """
    
    since = time.time()

    best_acc = 0.0
    best_loss=999999
    best_epoch=-1
    last_train_acc=-1
    last_train_loss=-1

    for epoch in range(start_epoch,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        #small_data flag used to decide on how to decay
        small_data=False
        if dataset_sizes['train']<=10000: small_data=True

        iter_at_lr=0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            i=0
            total_done=0
            for data in dataloaders[phase]:
                i+=1
                # get the inputs
                inputs, labels = data               
                batch_size= inputs.shape[0]
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                #needed for multilabel training which uses different loss and expects floats
                if not MULTICLASS:
                    labels = labels.float()
                    
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)
           
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                if MULTICLASS: # need to fix this for multilabel
                    running_corrects += torch.sum(preds == labels.long().data)
                running_loss += loss.data[0]*batch_size                                                 

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase=='train':
                last_train_acc=epoch_acc
                last_train_loss=epoch_loss

            print(phase+' epoch {}:loss {:.4f} acc: {:.4f} with data size {}'.format(
                epoch, epoch_loss, epoch_acc, dataset_sizes[phase]))

            #decay if not best
            if phase == 'val' and epoch_loss > best_loss:
                #normally we just decay if no improvement in val loss in epoch. 
                #but this is not good with small datasets
                #so I have this 'small_data' condition that insists on 5 passes at lr if dataset size <=10k
                if small_data==False or iter_at_lr>=4:
                    print("decay loss from "+str(LR)+" to "+str(LR/10)+" as not seeing improvement in val loss")
                    LR = LR / 10
                    #making a new optimizer works better as it zeros out momentum; just changing LR and keeping old momentum worked less well
                    optimizer = optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr = LR, momentum=0.9, weight_decay=1e-4)
                    iter_at_lr=0
                else:
                    iter_at_lr+=1

            #below is used for labeling results
            trainstring = str(TRAIN_FILTER).replace("_","").replace("[","").replace(",","_").replace("]","").replace(" ","").replace("'","")
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                #save stuff if we have a best model
                write_label = str(PRED_LABEL)
                write_label = "Multilabel"
                checkpoint(model, best_acc, best_loss, epoch, RESULT_PATH+write_label+"_train_"+trainstring+"_"+FOLD_OVERRIDE,LR,RESULT_PATH=RESULT_PATH)
            
            write_label = "multilabel_" + trainstring + "_" + FOLD_OVERRIDE
            if phase== 'val':
                with open(RESULT_PATH+"log_train_"+write_label, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([write_label, epoch, last_train_loss, last_train_acc, epoch_loss, epoch_acc])
            
        total_done+=batch_size
        if(total_done % (100*batch_size) == 0): print("completed "+str(total_done)+" so far in epoch")
        #quit if 3 epochs no improvement
        if ((epoch-best_epoch)>=3 and small_data==False) or ((epoch-best_epoch)>=15 and small_data==True): 
            print("no improvement in 3 epochs, break")
            break
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights and return them
    checkpoint_best = torch.load(RESULT_PATH+"checkpoint_results/Multilabel_train_"+trainstring+"_"+FOLD_OVERRIDE)
    model = checkpoint_best['model_ft']
    return model, best_epoch

def give_mean_var(LABEL_PATH, PRED_LABEL,BALANCE_MODE, TRAIN_FILTER,MULTILABEL, FOLD_OVERRIDE, BATCH_SIZE):
    """
    args:
    LABEL_PATH: path to the scalars file
    PRED_LABEL: list of targets we're predicting
    BALANCE_MODE: deprecated
    TRAIN_FILTER: list of dataset we're training on, needed for dataloader
    MULTILABEL: deprecated, always true
    FOLD_OVERRIDE: train/val/test split column name in scalars
    BATCH_SIZE: passes batch for dataloader
    returns:
        mean: rgb channel means np array 3x1
        std:: rgb channel std np array 3x1
    """
           
    #create set of val transforms
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),     
        transforms.Scale(224),
        transforms.CenterCrop(224), #needed to get 224x224
        transforms.ToTensor()
    ])

    #make dataloader
    transformed_dataset =CXR.CXRDataset(csv_file=LABEL_PATH, fold='train', PRED_LABEL=PRED_LABEL, transform=data_transform, balance_classes=BALANCE_MODE, FILTER=TRAIN_FILTER,MULTILABEL=MULTILABEL,FOLD_OVERRIDE=FOLD_OVERRIDE,SAMPLE=0,TRAIN_FILTER=TRAIN_FILTER,RESULT_PATH="ignore",MULTICLASS=MULTICLASS)
    
    dataloader = torch.utils.data.DataLoader(transformed_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=8)#, sampler=sampler)
           
    #calculate some means and st devs
           
    x = len(dataloader)*BATCH_SIZE
    print("len dataloader for give_mean_var:"+str(x))
    means = np.empty((x,3))
    stds =  np.empty((x,3))
    means[:,:]=np.nan
    stds[:,:]=np.nan
    
    for data in dataloader:
        inputs, _ = data
        inputs=inputs.numpy()
        for i in range(0,inputs.shape[0]):
            for j in range(0,3):
                means[i,j]=np.mean(inputs[i,j,:,:])
                stds[i,j]=np.std(inputs[i,j,:,:])
        
        
    mean = np.zeros(3)
    std = np.zeros(3)
    
    for j in range (0,3):
        x=np.nanmean(means[:,j])
        mean[j]=x
        x=np.nanmean(stds[:,j])
        std[j]=x

    return mean, std



def train_one(PRED_LABEL,LR,BATCH_SIZE,LABEL_PATH,RESULT_PATH,BALANCE_MODE,FREEZE_LAYERS, NUM_EPOCHS,TRAIN_FILTER,PRED_FILTER,MULTILABEL,FOLD_OVERRIDE,TRAIN_SAMPLE,PRED_SAMPLE,CUSTOM_NORMALIZE, NET_TYPE,MULTICLASS,OUTPUT1024):
    """ 
    make dataloader, instantiates torchvision model, calls training function, returns results
    
    args:  
         PRED_LABEL: list of labels to predict ["pna","opacity"] etc
         LR: learning rate
         BATCH_SIZE: batch size for dataloader; too big and won't fit on gpu
         LABEL_PATH: path to scalars
         RESULT_PATH: path to write results
         BALANCE_MODE:  deprecated
         FREEZE_LAYERS: deprecated
         NUM_EPOCHS:  max number of epochs to train for; may quit sooner if not improving
         TRAIN_FILTER: list of sites we're training on
         PRED_FILTER: list of sites we're predicting
         MULTILABEL: deprecated
         FOLD_OVERRIDE: train/val/test split column in scalars
         TRAIN_SAMPLE: sample training data to get limited sample (for testing)
         PRED_SAMPLE: sample test data to get limited sample (for testing)
         CUSTOM_NORMALIZE: use normalization mean, std based on data not imagenet
         NET_TYPE: deprecated
         MULTICLASS: train to single multiclass n>2 target (not implemented for multilabel multiclass)
         
    returns:
        x: df with predictions
    """
    #if we were using custom normalization and not imagenet, do this; it didn't help vs imagenet nornmalization
    if CUSTOM_NORMALIZE:
            mean, std = give_mean_var(LABEL_PATH, PRED_LABEL,BALANCE_MODE, TRAIN_FILTER,MULTILABEL, FOLD_OVERRIDE, BATCH_SIZE)
            print(mean)
            print(std)
    elif not CUSTOM_NORMALIZE:
            mean= [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
    
    #torchvision transforms
    df = pd.read_csv(LABEL_PATH,index_col=0)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),         
            transforms.Scale(224), #244
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
           
    #make dataloader       

    transformed_datasets={}
    transformed_datasets['train'] =CXR.CXRDataset(csv_file=LABEL_PATH, fold='train', PRED_LABEL=PRED_LABEL, transform=data_transforms['train'], balance_classes=BALANCE_MODE, FILTER=TRAIN_FILTER,MULTILABEL=MULTILABEL,FOLD_OVERRIDE=FOLD_OVERRIDE,SAMPLE=TRAIN_SAMPLE,TRAIN_FILTER=TRAIN_FILTER,RESULT_PATH=RESULT_PATH,MULTICLASS=MULTICLASS)
    transformed_datasets['val'] =CXR.CXRDataset(csv_file=LABEL_PATH, fold='val', PRED_LABEL=PRED_LABEL, transform=data_transforms['val'], balance_classes=BALANCE_MODE, FILTER=TRAIN_FILTER,MULTILABEL=MULTILABEL,FOLD_OVERRIDE=FOLD_OVERRIDE,SAMPLE=TRAIN_SAMPLE,TRAIN_FILTER=TRAIN_FILTER,RESULT_PATH=RESULT_PATH,MULTICLASS=MULTICLASS)
    dataloaders={}
    dataloaders['train'] = torch.utils.data.DataLoader(transformed_datasets['train'], batch_size=BATCH_SIZE, shuffle=True,num_workers=8)#, sampler=sampler)
    dataloaders['val'] = torch.utils.data.DataLoader(transformed_datasets['val'], batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    
    
    #instantiate model      
    if not use_gpu: raise ValueError("Error, requires GPU")
    print('==> Building model..')
                      
    if(NET_TYPE=="densenet121"):
        print("using densenet121")
        model_ft = models.densenet121(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        if(OUTPUT1024==False):
            print("adding bottleneck=15 features")
            #if multiclass, needs different output structure then regular training to list of binary taragets
            if not MULTICLASS:
                model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, 15), nn.Linear(15, len(PRED_LABEL)),nn.Sigmoid())
            elif MULTICLASS:
                model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, 15), nn.Linear(15, transformed_datasets['train'].n_class))
                print("n_class "+str(transformed_datasets['train'].n_class))
        elif(OUTPUT1024==True):
            print("NOT adding bottleneck=15 features")
            #if multiclass, needs different output structure then regular training to list of binary taragets
            if not MULTICLASS:
                model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, len(PRED_LABEL)),nn.Sigmoid())
            elif MULTICLASS:
                model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, transformed_datasets['train'].n_class))
                print("n_class "+str(transformed_datasets['train'].n_class))




    start_epoch = 1
    print("loading model_ft onto gpu")
    model_ft = model_ft.cuda()
               
    if NET_TYPE=="densenet121":
        if(MULTICLASS==False):
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss() # only using this for predicting site, department

    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=LR, momentum=0.9, weight_decay=1e-4)

                    
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    #train
    model_ft , best_epoch = train_model(model_ft, criterion, optimizer_ft, LR, num_epochs=NUM_EPOCHS,dataloaders=dataloaders,dataset_sizes=dataset_sizes, PRED_LABEL=PRED_LABEL, start_epoch=start_epoch,MULTILABEL=MULTILABEL,FOLD_OVERRIDE=FOLD_OVERRIDE,TRAIN_FILTER=TRAIN_FILTER,RESULT_PATH=RESULT_PATH,MULTICLASS=MULTICLASS)

    #make preds on test
    x = E.make_pred_multilabel(data_transforms,model_ft,"pred_"+str(PRED_LABEL), LABEL_PATH,RESULT_PATH,PRED_LABEL,TRAIN_FILTER,PRED_FILTER,FOLD_OVERRIDE,PRED_SAMPLE,MULTICLASS,OUTPUT1024)

    return x
    
    
    
    
def train_cnn(LABEL_PATH, PRED_LABEL,TRAIN_FILTER,PRED_FILTER,BALANCE_MODE,FOLD_OVERRIDE,MULTICLASS=False,OUTPUT1024=False):
    """
    main function that gets called externally to train
        LABEL_PATH: path to scalars
        PRED_LABEL: targets to predict; list ["pna","opacity"] etc as in scalars file
        TRAIN_FILTER: list of sites we're training to ["nih","msh"]
        PRED_FILTER: list of sites we're predicting ["nih","iu"]
        BALANCE_MODE: deprecated
        FOLD_OVERRIDE: the column of scalars we use for train val test split
        MULTICLASS: train to single multiclass n>2 target
    returns:
        y: results
    """
    NUM_EPOCHS=50
    BATCH_SIZE=16
    LR = 0.01 
    RESULT_PATH="results/"
    FREEZE_LAYERS="no"
    MULTILABEL = not isinstance(PRED_LABEL, str)
    TRAIN_SAMPLE=0
    PRED_SAMPLE =0
    CUSTOM_NORMALIZE=False
    NET_TYPE="densenet121"
    
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    if not os.path.exists(RESULT_PATH+"checkpoint_results/"):
        os.makedirs(RESULT_PATH+"checkpoint_results/")
    
    x = train_one(PRED_LABEL,LR,BATCH_SIZE,LABEL_PATH,RESULT_PATH,BALANCE_MODE,"layer4",NUM_EPOCHS,TRAIN_FILTER,PRED_FILTER,MULTILABEL,FOLD_OVERRIDE,TRAIN_SAMPLE,PRED_SAMPLE,CUSTOM_NORMALIZE, NET_TYPE, MULTICLASS,OUTPUT1024)
    
    y = pd.read_csv(LABEL_PATH)
    y=y[['img_id']]
    y = y.merge(x,on="img_id",how="inner")
    trainlist=str(TRAIN_FILTER).replace("_","").replace("[","").replace(",","_").replace("]","").replace(" ","").replace("'","")    
    y.to_csv(RESULT_PATH+"preds_train_"+trainlist+"_"+FOLD_OVERRIDE+".csv",index=False)
    return y







                
