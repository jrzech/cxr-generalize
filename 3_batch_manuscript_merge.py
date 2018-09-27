#main experiment

path = "manu-main-experiment/"

import pandas as pd
import numpy as np
import re
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
import sklearn.metrics as sklm
import sys
from scipy.special import logit, expit

#get baseline prevalences for data
baseline = pd.read_csv("data/scalars.csv")

try:
    rollup
    del rollup
except:
    pass


for group in ["msh_nih","msh","nih"]:
    df1=pd.read_csv(path+"preds_train_"+group+"_sync_fold.csv")
    try:
        rollup = rollup.merge(df1,on="img_id", how = "inner")
        print(len(rollup))
    except NameError:
        rollup = df1
        print(len(rollup))
        
rollup['group']=rollup['img_id'].str[0:3].str.replace("_","")

ground_truth = pd.read_csv("data/scalars.csv")
#ground_truth = ground_truth[ground_truth['view_use']==1]
ground_truth = ground_truth[['img_id','Pneumonia']]

rollup = rollup.merge(ground_truth,on="img_id",how="inner")

i=1
auc = pd.DataFrame(columns=["pred_group","train_group","label","auc"])

for pred_group in ["nih","msh","iu","all"]:
    for column in rollup.columns:
        if column == "img_id": continue
        if column == "group": continue
        if not "prob_Pneumonia" in column: continue
        print("do "+column) 
        x={}
        x['pred_group']=pred_group
        x['label']=column.replace("prob_","").replace("_msh","").replace("_iu","").replace("_nih","").replace("_train","")
        x['train_group']=column.replace(x['label'],"").replace("prob_","").replace("_train_","")
        
        if pred_group=="all":
            actual=rollup[x['label']].as_matrix().astype(int)
            pred = rollup[column].as_matrix()        
        else:        
            actual=rollup[rollup['group']==pred_group][x['label']].as_matrix().astype(int)
            pred = rollup[rollup['group']==pred_group][column].as_matrix()
            
        #alt_prob
        ap = pd.DataFrame()
        ap['img_id']=rollup['img_id']
        ap['train_group']=x['train_group']
        ap['pred_group']=rollup['group']
        ap['label']=x['label']
        ap['pred']=rollup[column]
        ap['actual']=rollup[x['label']]
        ap=ap[ap['pred_group']==x['pred_group']]
        
        try:
            alt_prob = pd.concat([alt_prob,ap])
        except:
            alt_prob = ap
            
        
        x['auc']=sklm.roc_auc_score(actual, pred)
        x['len']=len(actual)
        auc = auc.append(x,ignore_index=True)
        i+=1

auc.to_csv(path+"auc_results.csv",index=False)   
rollup.to_csv(path+"rollup_probs.csv",index=False)   
alt_prob.to_csv(path+"rollup_probs_nopivot.csv",index=False)   

print("mean auc = "+str(auc.mean()))

#imbalance experiment

path = "manu-imbalance-experiment/"
#get baseline prevalences for data
baseline = pd.read_csv("data/scalars.csv")

del rollup
del auc
del alt_prob

for group in ["balanced","msh_mild","msh_severe","nih_mild","nih_severe"]:
    df1=pd.read_csv(path+"preds_train_msh_nih_"+group+".csv")
    #print(df1.columns)
    recol = [x+"_bal_"+group for x in df1.columns]
    recol[0]="img_id"
    df1.columns=recol
    try:
        rollup = rollup.merge(df1,on="img_id", how = "outer")
        print("len after inner join on rollup")
        print(len(rollup))
    except NameError:
        rollup = df1
        print("len after init set of rollup")
        print(len(rollup))

####
rollup['group']=rollup['img_id'].str[0:3].str.replace("_","")

ground_truth = pd.read_csv("data/scalars.csv")
ground_truth = ground_truth[['img_id','Pneumonia']]
print(ground_truth.shape)

rollup = rollup.merge(ground_truth,on="img_id",how="inner")
print("rollup dim before analysis")
print(len(rollup))

i=1
auc = pd.DataFrame(columns=["pred_group","train_group","label","auc"])

for pred_group in ["nihmsh","nih","msh","iu"]:
    for column in rollup.columns:
        if column == "img_id": continue
        if column == "group": continue
        if not "prob_Pneumonia" in column: continue
        print("do column "+column+" for pred group "+pred_group)


        x={}
        x['pred_group']=pred_group
        x['label']=column.replace("prob_","").replace("_msh","").replace("_iu","").replace("_nih","").replace("_train","").replace("_bal_balanced","").replace("_bal_mild","").replace("_bal_severe","")
        x['train_group']=column.replace(x['label'],"").replace("prob_","").replace("_train_","")

        #since we have to do outer join with diff ids, need to now filter to ones with non-nan preds
        filtered = rollup[~rollup[column].isnull()][[x['label'],column,'group','img_id']]
        #print(len(filtered))
        #print("about to consider pred_group all")
        if pred_group=="all":
            actual=filtered[x['label']].as_matrix().astype(int)
            pred = filtered[column].as_matrix()
        elif pred_group=="nihmsh":
            actual=filtered[(filtered['group']=="nih") | (filtered['group']=="msh")][x['label']].as_matrix().astype(int)
            pred = filtered[(filtered['group']=="nih") | (filtered['group']=="msh")][column].as_matrix()
        else:
            actual=filtered[filtered['group']==pred_group][x['label']].as_matrix().astype(int)
            pred = filtered[filtered['group']==pred_group][column].as_matrix()
        #print(len(actual))
        #print(len(pred))
        #alt_prob
        #print("about to create ap")
                                        
        ap = pd.DataFrame()
        ap['img_id']=filtered['img_id']
        ap['train_group']=x['train_group']
        ap['pred_group']=filtered['group']
        #print("did group")
        ap['label']=x['label']
        #print("did label")
        ap['pred']=filtered[column]
        #print("did pred, see this x['label']")
        #print(rollup.columns)

        #print(rollup[x['label']])
        ap['actual']=filtered[x['label']]
        #print("did actual")
        ap=ap[ap['pred_group']==x['pred_group']]
        #print("made ap")
        try:
            alt_prob = pd.concat([alt_prob,ap])
        except:
            alt_prob = ap

        try:
            x['auc']=sklm.roc_auc_score(actual, pred)
        except:
            print("auc nan for "+column)
            x['auc']=np.nan

        x['len_test']=len(pred)
        auc = auc.append(x,ignore_index=True)
        i+=1

auc.to_csv(path+"auc_results.csv",index=False)
rollup.to_csv(path+"rollup_probs.csv",index=False)
alt_prob.to_csv(path+"rollup_probs_nopivot.csv",index=False)

print("mean auc = "+str(auc.mean()))

import os
if not os.path.exists("manu-results"):
    os.makedirs("manu-results")
