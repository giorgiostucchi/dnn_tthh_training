import numpy
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import keras
import mplhep as hep
import uproot
import uproot3
import json
import h5py

LR = [0.005, 0.01]
BS = [64, 128, 256]
NODE = [12, 24]
LAYER = [3, 4]
DROPOUT = [0.1, 0.2, 0.3]

auc_dict = []
auc = []
# compare validation AUC at last epoch
for lr in LR:
    for bs in BS:
        for node in NODE:
            for layer in LAYER:
                for dropout in DROPOUT:
                    history = "20230501_pTreco_off/hyperparameter_scan/LR" + str(lr) + "_BS" + str(bs) + "_node" + str(node) + "_layer"+ str(layer) + "_dropout"+str(dropout) + "_even/history.json"
                    with open(history, 'r') as history_file:
                        history_dict = json.load(history_file)
                        auc_last_epoch = history_dict["val_auc"]["19"]
                        auc.append(auc_last_epoch)
                        auc_dict.append({"LR": lr, "batchsize": bs, "node": node, "layer": layer, "dropout":dropout, "auc_last_epoch": auc_last_epoch})
        
print(auc_dict[numpy.argmax(auc)])

