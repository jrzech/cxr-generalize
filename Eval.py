import torch
import pandas as pd
import CXRDataset as CXR
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np
import math
from copy import deepcopy
import torch.nn as nn

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def make_pred_multilabel(data_transforms,model_ft,infostring,LABEL_PATH,RESULT_PATH, PRED_LABEL,TRAIN_FILTER,PRED_FILTER,FOLD_OVERRIDE,PRED_SAMPLE,MULTICLASS,OUTPUT1024):
    """
    args:
        data_transforms: torchvision transforms for dataloader
        model_ft: fine tuned model
        infostring: string for labeling results
        LABEL_PATH: path to scalars
        RESULT_PATH: path to results
        PRED_LABEL: labels that model predicts; index maps to output from model_ft
        TRAIN_FILTER: list of sites we're training to
        PRED_FILTER: list of sites we're predicting on
        FOLD_OVERRIDE: column of scalars with train/val/test split
        PRED_SAMPLE: filter to smaller # of data to predict; for testing only
        MULTICLASS: are we predicting single multiclass n>2 target, not a list of binary targets?
    returns: 
        return_pd: dataframe with predictions
    """

    model_ft.train(False) # needed for batchnorm
    
    model_ex = deepcopy(model_ft)
    if not MULTICLASS:
        new_fc = nn.Sequential(*list(model_ex.classifier.children())[:-2])
    elif MULTICLASS:
        new_fc = nn.Sequential(*list(model_ex.classifier.children())[:-1])
    model_ex.classifier = new_fc
            
    readme = pd.read_csv(LABEL_PATH)
    if 'view_impute' in readme:
        readme.rename(columns={'view_impute':'view_use'},inplace=True)
    readme=readme[readme['view_use']==1]
    
    thisfold="test"
    transformed_datasets_all =CXR.CXRDataset(csv_file=LABEL_PATH, fold=thisfold, PRED_LABEL=PRED_LABEL, transform=data_transforms['val'], balance_classes=False, FILTER=PRED_FILTER,MULTILABEL=True,FOLD_OVERRIDE=FOLD_OVERRIDE,SAMPLE=PRED_SAMPLE,TRAIN_FILTER=TRAIN_FILTER,RESULT_PATH=RESULT_PATH,MULTICLASS=MULTICLASS)
        
    BATCH_SIZE=16
    dataloaders_all= torch.utils.data.DataLoader(transformed_datasets_all, BATCH_SIZE, shuffle=False, num_workers=8)
    size = len(transformed_datasets_all)
    return_pd = pd.DataFrame(columns=["img_id"])
    true_pd = pd.DataFrame(columns=["img_id"])
    sigmoid_v = np.vectorize(sigmoid)
    trainlist=str(TRAIN_FILTER).replace("_","").replace("[","").replace(",","_").replace("]","").replace(" ","").replace("'","")
    pd.DataFrame(data=transformed_datasets_all.df['img_id']).to_csv(RESULT_PATH+"last_layer_features"+trainlist+"_"+FOLD_OVERRIDE+"_img_ids.csv",index=False, header=False)
    correct=0
    for i, data in enumerate(dataloaders_all):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model_ex(inputs)        
        y=outputs.cpu().data.numpy()
        true_labels=labels.cpu().data.numpy()
        
        batch_size=y.shape
        bn_dim=15
        if(OUTPUT1024): bn_dim=1024
        fc_matrix = np.zeros((int(batch_size[0]),bn_dim))
        fc_matrix[0:batch_size[0],:]=y
        
        outputs = model_ft(inputs)        
        y=outputs.cpu().data.numpy()
        probs = y
        for j in range(0,batch_size[0]):            
            #get last layer
            thisrow={}
            truerow={}
            thisrow["img_id"]=transformed_datasets_all.df['img_id'].iloc[BATCH_SIZE*i+j]
            truerow["img_id"]=transformed_datasets_all.df['img_id'].iloc[BATCH_SIZE*i+j]
            
            if not MULTICLASS:
                for k in range(len(PRED_LABEL)):
                    thisrow["prob_"+PRED_LABEL[k]+"_train_"+trainlist]=probs[j,k]
                    truerow["prob_"+PRED_LABEL[k]+"_train_"+trainlist]=true_labels[j,k]
            elif MULTICLASS:
                    _, predx = torch.max(outputs.data, 1) #
                    thisrow["pred_"+PRED_LABEL[0]+"_train_"+trainlist]=transformed_datasets_all.int_to_label[predx[j]]
                    truerow["pred_"+PRED_LABEL[0]+"_train_"+trainlist]=transformed_datasets_all.int_to_label[true_labels[j]]
                    if thisrow["pred_"+PRED_LABEL[0]+"_train_"+trainlist]==truerow["pred_"+PRED_LABEL[0]+"_train_"+trainlist]: correct+=1
                
            return_pd=return_pd.append(thisrow,ignore_index=True)
            true_pd = true_pd.append(truerow,ignore_index=True)
            
        #write last layer
        f=open(RESULT_PATH+"last_layer_features"+trainlist+"_"+FOLD_OVERRIDE+".csv",'ab')        
        np.savetxt(f,fc_matrix, fmt='%10.12f',delimiter=",",)
        f.close()
            
        if(i%10==0): print(str(i*16))
      
    if MULTICLASS:
        print("accuracy:"+str(correct/len(return_pd)))
        return_pd['ground_truth']=true_pd["pred_"+PRED_LABEL[0]+"_train_"+trainlist]
    elif not MULTICLASS:
        auc_num = 0
        auc_denom = 0
        for column in return_pd:
            if column == "img_id": continue
            print(column+" auc :")
            actual=true_pd[column]
            pred=return_pd[column]
            print(actual.sum())
            auc_score=np.nan
            try:
                auc_score=sklm.roc_auc_score(actual.as_matrix().astype(int), pred.as_matrix())
            except:
                print("couldn't calculate auc for "+column)
            auc_denom+=1
            auc_num+=auc_score

        print("average auc:"+str(auc_num/auc_denom))
            
    return return_pd
