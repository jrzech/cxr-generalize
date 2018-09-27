# cxr-generalize

Code used to evaluate generalization performance of CNN models to detect pneumonia described in ["Confounding variables can degrade generalization performance of radiological deep learning models."](https://arxiv.org/abs/1807.00431)

## Introduction

For those who are new to this work, I strongly suggest starting with the [reproduce-chexnet repo](https://github.com/jrzech/reproduce-chexnet), which is specifically designed to promote ease of use and reproducibility. It allows you to run code in the browser to get started with no local configuration needed:

[![Illustration](https://www.github.com/jrzech/reproduce-chexnet/raw/master/illustration.png?raw=true "Illustration")](https://github.com/jrzech/reproduce-chexnet)

It also contains instructions to allow you to quickly clone the repo and reproduce needed dependencies on your system using anaconda in a streamlined way. It is a good place to get started with this work and can serve as a stepping-stone to get your own independent projects based on CNNs started.

## This repo

The code in this cxr-generalize repo corresponds to a related research project evaluating how well such models generalized between three different hospitals. Unfortunately, due to restrictions on the data, the code on this repo cannot be run out of the box, and requires local configuration, data acquisition, and labeling.

## Data download

Three datasets were used in this research project: data from NIH, Indiana University, and Mount Sinai Hospital. NIH labels are included with this (`data/scalars.csv`). The NIH chest x-rays themselves are large (>40 gb) and must be [downloaded separately](https://nihcc.app.box.com/v/ChestXray-NIHCC). The labels we derived from IU cannot be shared under the terms of their licensing agreement which prohibits derivative works, but the x-rays and labels can be [freely downloaded](https://openi.nlm.nih.gov/faq.php?it=xg) and are straightforward to merge with the NIH labels we provide. The dataset we used from Mount Sinai Hospital cannot be shared under the terms of our IRB approval.

## Additional labeling

To use this code to assess generalization performance of CNNs, you will need to add additional label data from sites you wish to compare against to `scalars.csv`, and then modify the included code to reflect the site codes on which you are training (in this code, it assumes NIH, Indiana, and Mount Sinai - so ["nih", "iu", "msh"]).

## Code available

The `RUN_ALL.sh` script runs all model training and evaluation code; be sure previous result folders are cleared/deleted before running. `1_batch...` to `5_batch...py`: these are run sequentially, 1-5, to generate results; `RUN_ALL.sh` runs them all.

Information on output generated:

Each results folder has individual output files from models:
- `last_layer`refers to bottleneck features
- `log_train`gives training history
- `preds` gives predictions on test data for each model
- `bottleneck.csv` gives activations from the final bottleneck layer
- `rollup_probs.csv` and `rollup_probs_nopivot.csv` files gives aggregated probabilities of pathology for each image in test 

Individual files containing underlying code:
- `CXRDataset.py`: Dataset used to load CXR
- `Eval.py`: code used for model evaluation
- `model.py`: core model training code



