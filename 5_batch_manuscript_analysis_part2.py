#calc 
import pandas as pd
import numpy as np
import os

if not os.path.exists("manu-results"):
    os.makedirs("manu-results")


hospital=pd.read_csv("manu-hospital/preds_train_nih_msh_iu_site_fold.csv")
department=pd.read_csv("manu-department/preds_train_msh_site_fold.csv")

#print(hospital.head())
hospital_acc=hospital.groupby(['ground_truth','pred_group_train_nih_msh_iu']).count()
hospital_acc['denom']=np.nan
hospital_acc['ground_truth']=hospital_acc.index.get_level_values(0)
for i in range(0,len(hospital_acc)):
    hospital_acc['denom'].iloc[i]=np.sum(hospital['ground_truth']==hospital_acc['ground_truth'].iloc[i])
hospital_acc['pct']=(hospital_acc['img_id'].T/hospital_acc['denom'].T).T
hospital_acc.to_csv("manu-results/hospital-accuracy.csv")

print(department.head())
department_acc=department.groupby(['ground_truth','pred_dicom_machine_binary_train_msh']).count()
department_acc['denom']=np.nan
department_acc['ground_truth']=department_acc.index.get_level_values(0)
for i in range(0,len(department_acc)):
    department_acc['denom'].iloc[i]=np.sum(department['ground_truth']==department_acc['ground_truth'].iloc[i])
department_acc['pct']=(department_acc['img_id'].T/department_acc['denom'].T).T
department_acc.to_csv("manu-results/department-accuracy.csv")
