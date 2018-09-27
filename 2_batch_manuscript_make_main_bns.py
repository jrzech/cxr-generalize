import pandas as pd
import numpy as np
pd.options.display.max_columns = 150
pd.options.display.max_rows = 150

try:
    del big_df
except:
    pass

path = "manu-main-experiment/"

for group in ['msh','msh_nih','nih']:
    df1=pd.read_csv(path+"last_layer_features"+group+"_sync_fold.csv",header=None)
    df2=pd.read_csv(path+"last_layer_features"+group+"_sync_fold_img_ids.csv",header=None)
    df1['img_id']=df2[0]
    df1['train_group']=group
    df1['pred_group']=df1['img_id'].str[0:3].replace("_","")
    
    try:
        big_df=pd.concat([big_df,df1])
    except:
        big_df=df1
 
cols = big_df.columns.tolist()
cols = cols[-3:] + cols[:-3]
big_df = big_df[cols]

print(len(big_df))
big_df.to_csv("manu-main-experiment/bottlenecks.csv",index=False)
