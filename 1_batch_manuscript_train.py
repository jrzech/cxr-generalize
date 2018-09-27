import pandas as pd
import numpy as np
pd.options.display.max_columns = 4000
import model as m
from importlib import reload
reload(m)
from copy import deepcopy
import os

#train imbalance
PRED_LABEL=["Cardiomegaly","Emphysema","Effusion","Hernia","Nodule","Atelectasis","Pneumonia","Edema","Consolidation"]
LABEL_PATH="data/scalars.csv"

for cv in ["balanced","msh_mild","msh_severe","nih_mild","nih_severe"]:
    return_df = m.train_cnn(LABEL_PATH = LABEL_PATH, PRED_LABEL=PRED_LABEL,TRAIN_FILTER=["msh","nih"],PRED_FILTER=["nih","iu","msh"], BALANCE_MODE="default",FOLD_OVERRIDE=cv)

os.rename('results','manu-imbalance-experiment')

#train main experiment

return_df = m.train_cnn(LABEL_PATH = LABEL_PATH, PRED_LABEL=PRED_LABEL,TRAIN_FILTER=["msh"],PRED_FILTER=["nih","iu","msh"], BALANCE_MODE="default",FOLD_OVERRIDE="sync_fold")

return_df = m.train_cnn(LABEL_PATH = LABEL_PATH, PRED_LABEL=PRED_LABEL,TRAIN_FILTER=["nih"],PRED_FILTER=["nih","iu","msh"], BALANCE_MODE="default",FOLD_OVERRIDE="sync_fold")

return_df = m.train_cnn(LABEL_PATH = LABEL_PATH, PRED_LABEL=PRED_LABEL,TRAIN_FILTER=["msh","nih"],PRED_FILTER=["nih","iu","msh"], BALANCE_MODE="default",FOLD_OVERRIDE="sync_fold")

os.rename('results','manu-main-experiment')

#predict department
PRED_LABEL=["dicom_machine_binary"]

return_df = m.train_cnn(LABEL_PATH = LABEL_PATH, PRED_LABEL=PRED_LABEL,TRAIN_FILTER=["msh"],PRED_FILTER=["msh"], BALANCE_MODE="default",FOLD_OVERRIDE='site_fold',MULTICLASS=True)
os.rename('results','manu-department')

#predict hospital system

PRED_LABEL=["group"]

return_df = m.train_cnn(LABEL_PATH = LABEL_PATH, PRED_LABEL=PRED_LABEL,TRAIN_FILTER=["nih","msh","iu"],PRED_FILTER=["nih","msh","iu"], BALANCE_MODE="default",FOLD_OVERRIDE='site_fold',MULTICLASS=True)

os.rename('results','manu-hospital')

#predict hospital system 1024 output

PRED_LABEL=["group"]

return_df = m.train_cnn(LABEL_PATH = LABEL_PATH, PRED_LABEL=PRED_LABEL,TRAIN_FILTER=["nih","msh","iu"],PRED_FILTER=["nih","msh","iu"], BALANCE_MODE="default",FOLD_OVERRIDE='site_fold',MULTICLASS=True,OUTPUT1024=True)

os.rename('results','manu-hospital-1024')


print("training complete")





