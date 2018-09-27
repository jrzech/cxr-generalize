python 1_batch_manuscript_train.py
python 2_batch_manuscript_make_main_bns.py
python 3_batch_manuscript_merge.py
R CMD BATCH 4_batch_manuscript_analysis.R
python 5_batch_manuscript_analysis_part2.py

mv 4_batch_manuscript_analysis.Rout manu-results
mkdir manu-results/subanalysis
mkdir manu-results/subanalysis/site
mkdir manu-results/subanalysis/location_msh

cp manu-main-experiment/rollup_probs_nopivot.csv manu-results/main-probs.csv
cp manu-main-experiment/bottlenecks.csv manu-results/main-bottlenecks.csv
cp manu-hospital/preds_train_nih_msh_iu_site_fold.csv manu-results/subanalysis/site/preds_train_nih_msh_iu_site_fold.csv
cp manu-department/preds_train_msh_site_fold.csv manu-results/subanalysis/location_msh/preds_train_msh_site_fold.csv
cp manu-imbalance-experiment/rollup_probs_nopivot.csv manu-results/engineered_probs.csv



 
