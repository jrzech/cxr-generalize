import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image

class CXRDataset(Dataset):
    """OpenI dataset."""

    def __init__(self, csv_file, fold, PRED_LABEL, transform=None, balance_classes=False, FILTER=["iu","msh","nih"],MULTILABEL=True,FOLD_OVERRIDE=None,SAMPLE=0,TRAIN_FILTER="x",RESULT_PATH="results/",MULTICLASS=False):
        """
        initiates class to store cxr labels and filepaths, to be fed into dataloader for pytorch / torchvision

        args:
            csv_file: path to scalars
            fold: "train, test, val, pred" will be filtered to that value in FOLD_OVERRIDE; anything else ignored
            PRED_LABEL: targets we're predicting
            transform: torchvision transform
            balance_classes: deprecated
            FILTER: filter dataset to only these sites
            MULTILABEL: deprecated, always True
            FOLD_OVERRIDE: column in scalars with fold info
            SAMPLE: sample some number for testing only; if 0, takes all
            TRAIN_FILTER: for labeling result files
            RESULT_PATH: path for results
            MULTICLASS: true if predicting single target with n>2 classes, otherwise false
        returns:
            nothing

        """
        self.df = pd.read_csv(csv_file)
        self.df=self.df.set_index("revised_filepath")
        self.PRED_LABEL = PRED_LABEL
        self.MULTILABEL = MULTILABEL
        self.MULTICLASS = MULTICLASS
        if(MULTICLASS):
            self.label_to_int={}
            self.int_to_label={}
            self.n_class=0
        
        fold_col = FOLD_OVERRIDE
        print("note: override filtering fold on col "+FOLD_OVERRIDE)
        print("df length before fold filter is "+str(self.df.shape[0]))
        
        self.transform = transform
        if(fold=="train"):
            self.df=self.df[self.df[fold_col]=="train"]
        if(fold=="test"):
            self.df=self.df[self.df[fold_col]=="test"]
        if(fold=="val"):
            self.df=self.df[self.df[fold_col]=="val"]
        if(fold=="pred"):
            self.df=self.df[self.df[fold_col]=="pred"]
        print("df length after fold filter is "+str(self.df.shape[0]))
            
        self.df=self.df[self.df['group'].isin(FILTER)]

        if 'view_impute' in self.df.columns:
            self.df.rename(columns={'view_impute':'view_use'},inplace=True)

        self.df=self.df[self.df["view_use"]==1]

        print("df length after view_use filter is "+str(self.df.shape[0]))        
        
        #make sure valid data if we're in (train, val, test)
        if(fold=="train" or fold=="val" or fold=="test"):
            if not MULTICLASS:
                if not MULTILABEL:
                    self.df=self.df[(self.df[PRED_LABEL]==0) | (self.df[PRED_LABEL]==1 )]
                elif MULTILABEL:
                    for label in PRED_LABEL:
                        self.df=self.df[(self.df[label]==bool(0)) | (self.df[label]==bool(1)) ]
            elif MULTICLASS:
                print("multiclass len before remove nan")
                print(str(len(self.df)))
                self.df=self.df[~(self.df[PRED_LABEL[0]].isnull())]
                print("multiclass len after remove nan")
                print(str(len(self.df)))
                self.df[str(self.PRED_LABEL[0]+"_orig")]=self.df[self.PRED_LABEL[0]]             
                i=0
                gb = sorted(list(set(self.df[PRED_LABEL[0]])))
                print("gb")
                print(gb)
                iterdict=0
                for entry in gb:
                    self.label_to_int[entry]=iterdict
                    self.int_to_label[iterdict]=entry
                    print("i="+str(iterdict)+" corresponds to entry "+str(entry))
                    self.df[PRED_LABEL[0]]=np.where(self.df[PRED_LABEL[0]]==entry,iterdict,self.df[PRED_LABEL[0]])  
                    iterdict+=1
                self.n_class=iterdict
                print(self.df.groupby('group').count())        

        if(SAMPLE>0):
            self.df = self.df.sample(min(SAMPLE,len(self.df)))
            print("LIMITED TO SAMPLE OF "+str(SAMPLE)+" FOR TESTING")
       
        if(fold=="test"):
            trainlist=str(TRAIN_FILTER).replace("_","").replace("[","").replace(",","_").replace("]","").replace(" ","").replace("'","")    
            self.df['img_id'].to_csv(RESULT_PATH+"test_img_ids_train_"+trainlist+".csv",index=False,header=False)            
 
    def __len__(self):
        return len(self.df)                                              

    def __getitem__(self, idx):
        """
            used re return individual image, label pair
        """
        
        image = Image.open(self.df.index[idx])
        image = image.convert('RGB')
        
        if self.MULTILABEL == False:
            view = self.df[self.PRED_LABEL].iloc[idx].astype('int')
        elif self.MULTILABEL == True:
            if(self.MULTICLASS==False):
                view = np.zeros(len(self.PRED_LABEL),dtype=int)
                for i in range(0,len(self.PRED_LABEL)):
                   # print("about to look at column:" + self.PRED_LABEL[i].strip() + "*")
                    if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int')>0):
                        view[i]=self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int')
            elif(self.MULTICLASS==True):
                view = int(self.df[self.PRED_LABEL[0]].iloc[idx])
                

        if self.transform:
            image = self.transform(image)

        return (image,view)

