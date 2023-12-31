print("I am starting something here...")

import argparse
#from ROOT import *
import pandas as pd
import numpy
import uproot
import os
import seaborn as sn
import argparse
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from tensorflow import keras
#from keras.callbacks import History 
#history = History()
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from keras.regularizers import l1
#from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.activations as activations
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.losses as losses
#from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
# from sklearn_export import Export
import json
import shap
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import h5py
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay


def GetParser():
  """Argument parser for reading Ntuples script."""
  parser = argparse.ArgumentParser(
      description="Reading Ntuples command line options."
  )

  parser.add_argument(
      "--LR",
      type=float,
      help="specify the learning rate",
  )

  parser.add_argument(
      "--batchsize",
      type=int,
      help="specify the batchsize of the training",
  )

  parser.add_argument(
      "--node",
      type=int,
      help="specify the number of nodes in a hidden layer",
  )

  parser.add_argument(
      "--layer",
      type=int,
      help="specify the number of hidden layers",
  )

  parser.add_argument(
      "--dropout",
      type=float,
      help="specify the dropout rate",
  )

  args = parser.parse_args()
  return args

args = GetParser()

print("In the main now ...", flush=True)

DEFAULT = -999
LR = args.LR
BATCHSIZE = args.batchsize
EPOCH=10
nodes = [args.node]*args.layer
dropout_rate = args.dropout

#input_features = ["met_pt" ] #change these
#input_features = ["nrecojet_antikt4","nrecojet_antikt4_btag85","chi_hh","chi_hz","chi_zz","DeltaR_12", "DeltaR_34", "DeltaR_56", "DeltaPhi_12", "DeltaPhi_34","DeltaPhi_56", "DeltaEta_12", "DeltaEta_34","DeltaEta_56", "DeltaR_1234", "DeltaR_1256", "DeltaR_3456", "DeltaPhi_1234", "DeltaPhi_1256", "DeltaPhi_3456", "DeltaEta_1234", "DeltaEta_1256", "DeltaEta_3456", "DeltaEta_max", "DeltaEta_min", "DeltaEta_mean", "DeltaR_max", "DeltaR_min", "DeltaR_mean", "mH1_hh", "mH2_hh", "mH_hz", "mZ_hz", "mZ1_zz", "mZ2_zz", "m_jj56", "pT_1", 'pT_2', "pT_3", "pT_4", "pT_5", "pT_6", "eta_1", "eta_2", "eta_3", "eta_4", "eta_5", "eta_6", "pT_12", "pT_34", "pT_56", "sumHT_bjets", "sumHT_totalJets", "met_pt", "met_phi"]
'''
all input features:
'''
input_features = ["nrecojet_antikt4","nrecojet_antikt4_btag85","chi_hh","chi_hz","chi_zz","DeltaR_12", "DeltaR_34", "DeltaPhi_12", "DeltaPhi_34", "DeltaEta_12", "DeltaEta_34", "DeltaR_1234", "DeltaPhi_1234", "DeltaEta_1234", "DeltaEta_max", "DeltaEta_min", "DeltaEta_mean", "DeltaR_max", "DeltaR_min", "DeltaR_mean", "mH1_hh", "mH2_hh", "mH_hz", "mZ_hz", "mZ1_zz", "mZ2_zz","pT_1", 'pT_2', "pT_3", "pT_4", "eta_1", "eta_2", "eta_3", "eta_4", "pT_12", "pT_34", "sumHT_bjets", "sumHT_totalJets", "met_pt", "met_phi"]
'''
selected input features:
'''
#input_features = ["nrecojet_antikt4","nrecojet_antikt4_btag85","chi_hh","chi_zz","DeltaR_12", "DeltaR_34", "DeltaEta_max", "DeltaEta_min", "DeltaR_max", "DeltaR_mean", "mZ1_zz", "pT_1", 'pT_2', "pT_3", "pT_4","pT_12", "pT_34", "sumHT_bjets", "sumHT_totalJets", "met_pt"]
# Better function for saving keras predictions
def Save_pred(pred_train_sig,pred_train_bkg,pred_test_sig,pred_test_bkg, path):
  numpy.savetxt(str(path) + "/TrainSigPred.txt", pred_train_sig, delimiter=',')
  numpy.savetxt(str(path) + "/TrainBkgPred.txt", pred_train_bkg, delimiter=',')
  numpy.savetxt(str(path) + "/TestSigPred.txt", pred_test_sig, delimiter=',')
  numpy.savetxt(str(path) + "/TestBkgPred.txt", pred_test_bkg, delimiter=',')


######### pairing + classification model ###################
def create_model(input_dim):
  inputs = keras.Input(shape=(input_dim,))
  x = inputs
  x = layers.Masking(mask_value=DEFAULT)(x)
  for node in nodes:
    x = layers.Dense(node, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

  outclass = layers.Dense(1, activation="sigmoid")(x)

  model = keras.Model(inputs=inputs, outputs = outclass)

  model.compile(loss = 'binary_crossentropy'
  , optimizer = keras.optimizers.Adam(learning_rate=LR)
  , metrics=['Accuracy', 'AUC', 'FalseNegatives', 'FalsePositives', 'TrueNegatives', 'TruePositives']
  )
  
  return model
################################ LOADING INPUT DATASETS ################################

stop = -1    #with 1 M it works (with all variables), 2 M with 20 vrbls .... keep increasing, keeping just 20 variables

print("reading in background files", id)
bkg_file = uproot.open("/eos/user/g/gstucchi/NTRUPLES/isFullHad/background.root:MiniTree_NOSYS")
df_bkg = bkg_file.arrays(library="pd", entry_start=0, entry_stop=stop)
#print("applying cuts on number of jets")
#df_bkg = df_bkg[(df_bkg['nrecojet_antikt4'] >= 6) & (df_bkg['nrecojet_antikt4_btag85'] >= 6)]  #stricter cuts
# retrieve a single entry in df_bkg

#input_features = bkg_file.keys()
#input_to_remove = ["eventWeight","eventNumber","mX","rweight","rweightflat","rweightphi","ntruebjet","pthat","nrecob","nb77","sumPC","mcChannelNumber",
#                   "nHnotruthjetmatch","correctpair","hascorrectpair","nbquark"]

#input_features = [item for item in input_features if item not in input_to_remove]

columns = bkg_file.keys()
trainSigDF = pd.DataFrame(columns = columns)
trainBkgDF = pd.DataFrame(columns = columns)
testSigDF = pd.DataFrame(columns = columns)
testBkgDF = pd.DataFrame(columns = columns)
trainDF = pd.DataFrame(columns = columns)
testDF = pd.DataFrame(columns = columns)
train_sig_sample_wgts = []

num_train_sig = 0
num_test_sig = 0
print("reading in signal files", id)
sig_file = uproot.open("/eos/user/g/gstucchi/NTRUPLES/isFullHad/signal.root:MiniTree_NOSYS")
df_sig = sig_file.arrays(library="pd", entry_start=0, entry_stop=stop)
#print("applying cuts on number of jets")
#df_sig = df_sig[(df_sig['nrecojet_antikt4'] >= 6) & (df_sig['nrecojet_antikt4_btag85'] >= 6)]  #stricter cuts

# print the entries of nrecojet_antikt4
# print("nrecojet_antikt4 entries:", df_sig["nrecojet_antikt4"])

#uncomment if you want to plot the input features, they are the same every time
'''
#plot all the input_features specified above, keeping signal and background separated. the input features are the branches of the MiniTree
for feature in input_features:
  fig = plt.figure(figsize=(15,12))
  plt.style.use(hep.style.ROOT)
  hep.atlas.text(text='Internal', loc=1, fontsize=20)
  hep.atlas.text(text=r'$\sqrt{s}$=13 TeV, ttHH fullHad signal', loc=2, fontsize=20)
  plt.hist(df_sig[feature], bins=50, histtype="step", density=True, label="signal", color="steelblue")
  plt.hist(df_bkg[feature], bins=50, histtype="step", density=True, label="background", color="darkorange")
  plt.legend()
  ymin, ymax = plt.ylim()
  plt.ylim(ymin, ymax*1.2)
  plt.xlabel(feature)
  plt.ylabel("No. events")
  plot_name = "/afs/cern.ch/user/g/gstucchi/dnn_tthh_training/results/input_variables/" + feature + ".pdf"
  plt.ioff()
  fig.savefig(plot_name, transparent=True)
  plt.close(fig)

print("number of input features:", len(input_features))
# now do the same thing but plot everything as subfigures on the same plot
fig, axs = plt.subplots(6, 7, figsize=(90, 84))
plt.style.use(hep.style.ROOT)
hep.atlas.text(text='Internal', loc=1, fontsize=20)
hep.atlas.text(text=r'$\sqrt{s}$=13 TeV, ttHH fullHad signal', loc=2, fontsize=20)
for i, ax in enumerate(axs.flatten()):
  if i >= len(input_features):
    continue
  if i < len(input_features):  
    feature = input_features[i]
    ax.hist(df_sig[feature], bins=50, histtype="step", density=True, label="signal", color="steelblue")
    ax.hist(df_bkg[feature], bins=50, histtype="step", density=True, label="background", color="darkorange")
    ax.legend()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax*1.2)
    ax.set_xlabel(feature)
    ax.set_ylabel("No. events")
fig.tight_layout()
plot_name = "/afs/cern.ch/user/g/gstucchi/dnn_tthh_training/results/input_variables/all_variables.pdf"
plt.ioff()
fig.savefig(plot_name, transparent=True)
plt.close(fig)
'''

df_sig_train, df_sig_test = train_test_split(df_sig, test_size=0.5, shuffle= True)
# why is it splitting it here if we are doing the splitting afterwards as well with train_test_split?
trainSigDF = trainSigDF.append(df_sig_train, ignore_index=True)
print("Number of signal events in training:", len(trainSigDF))
testSigDF = testSigDF.append(df_sig_test, ignore_index=True)
print("Number of signal events in testing:", len(testSigDF))
train_sig_wgts = df_sig_train.loc[:, "mcEventWeight"].values # numpy.ones(len(df_sig_train))  #  provare con 1        
test_sig_wgts =  df_sig_test.loc[:, "mcEventWeight"].values   #  numpy.ones(len(df_sig_test))  # 
len_train_seg = len(df_sig_train)
len_test_seg = len(df_sig_test)

num_train_sig += len_train_seg
num_test_sig += len_test_seg
# weight event weights by number of events in each mass point
train_sig_sample_wgts.append(train_sig_wgts)

# then dividing the event weights by number of total signal events
train_sig_sample_wgts = numpy.concatenate(train_sig_sample_wgts).ravel()

# scaling the signal sample weights to have mean = 1 before adding in background sample weights
mean_sig_train_wgts = numpy.mean(train_sig_sample_wgts)
train_sig_sample_wgts += (1-mean_sig_train_wgts)
trainSigDF["sampleWeight"] = train_sig_sample_wgts

train_sig_y = [1]*num_train_sig
test_sig_y = [1]*num_test_sig

# split training/validation signal events
x_sig_train, x_sig_valid, y_sig_train, y_sig_valid = train_test_split(trainSigDF, train_sig_y, test_size=0.3, shuffle= True)

# Add background events and sample weights to training / testing sample

df_bkg_train, df_bkg_test = train_test_split(df_bkg, test_size=0.5, shuffle= True)

trainBkgDF = trainBkgDF.append(df_bkg_train, ignore_index=True)
print("Number of background events in training:", len(trainBkgDF))
testBkgDF = testBkgDF.append(df_bkg_test, ignore_index=True)
print("Number of background events in testing:", len(testBkgDF))
train_bkg_sample_wgts = df_bkg_train.loc[:, "mcEventWeight"].values #  numpy.ones(len(df_bkg_train)) # 
trainBkgDF["sampleWeight"] = train_bkg_sample_wgts

train_bkg_y = [0]*len(trainBkgDF)
test_bkg_y = [0]*len(testBkgDF)
# combining the targets 
train_y = numpy.append(train_sig_y, train_bkg_y)
test_y = numpy.append(test_sig_y, test_bkg_y)

# split training/validation background events
x_bkg_train, x_bkg_valid, y_bkg_train, y_bkg_valid = train_test_split(trainBkgDF, train_bkg_y, test_size=0.3, shuffle= True)

class_weight = {0: 1.,
                1: len(trainBkgDF)/len(trainSigDF)}

x_train = x_sig_train.append(x_bkg_train)
x_valid = x_sig_valid.append(x_bkg_valid)
y_train = numpy.append(y_sig_train, y_bkg_train)
y_valid = numpy.append(y_sig_valid, y_bkg_valid)
train_sample_wgts = x_train["sampleWeight"]
valid_sample_wgts = x_valid["sampleWeight"]
x_train = x_train.loc[:, input_features]
x_valid = x_valid.loc[:, input_features]

print("Number of signal events in training:", len(x_sig_train))
print("Number of background events in training:", len(x_bkg_train))
print("Number of signal events in validation:", len(x_sig_valid))
print("Number of background events in validation:", len(x_bkg_valid))
print("Signal yields in training", sum(x_sig_train["sampleWeight"]))

testDF = testDF.append(testSigDF, ignore_index=True)
testDF = testDF.append(testBkgDF, ignore_index=True)
y_test = numpy.append([1]*len(testSigDF), [0]*len(testBkgDF))
x_test = testDF.loc[:, input_features]

print("Training variables:", input_features)
nFeatures = len(input_features)
#eos
model_path = "/eos/user/g/gstucchi/dnn_tthh_training/results/LR" + str(args.LR) + "_BS" + str(args.batchsize) + "_node" + str(args.node) + "_layer"+ str(args.layer) + "_dropout"+str(args.dropout) + "_Evt" + str(stop) + "_Vrbl" + str(nFeatures) + "_Epochs" + str(EPOCH) + "/"
os.makedirs(model_path, exist_ok=True)
# Saving training variables to json
var_dict = {}
var_dict["variables"] = input_features
with open(model_path + "/variables.json", "w") as var_file:
  json.dump(var_dict, var_file)

########### Standardising training and validation data ###########
scaler = StandardScaler()
Scaling_train = scaler.fit(x_train)
x_train_scaled = Scaling_train.transform(x_train)
x_valid_scaled = Scaling_train.transform(x_valid)
x_test_scaled = Scaling_train.transform(x_test)
#scaling = Export(scaler)
#scale_dict = scaling.to_json()
#with open(model_path + "/scaling.json", "w") as outfile:
#  json.dump(scale_dict, outfile)


print("training")
model = create_model(nFeatures)
#model.summary()
history = model.fit(x_train_scaled
    , y_train
    , batch_size=BATCHSIZE
    , epochs=EPOCH
    , sample_weight = train_sample_wgts
    , class_weight=class_weight
    , validation_data=(x_valid_scaled, y_valid, valid_sample_wgts)
    )
json_string = model.to_json()
open(model_path + "/architecture.json", 'w').write(json_string)
model.save(model_path + '/model.h5', overwrite=True)
model.save_weights(model_path + "/weights.h5", overwrite=True)

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = model_path + '/history.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# plotting train/val loss
print("plotting train/val loss")
fig = plt.figure(figsize=(15,12))
plt.style.use(hep.style.ROOT)
hep.atlas.text(text='Internal', loc=1, fontsize=20)
hep.atlas.text(text=r'$\sqrt{s}$=13 TeV, ttHH fullHad signal', loc=2, fontsize=20)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'validation'], loc='upper right')
plt.ylabel('Loss')
plt.xlabel('Epoch')
name = model_path + "/TrainValLoss.pdf"
plt.ioff()
fig.savefig(name)
plt.close(fig)

train_sig = trainSigDF.loc[:,input_features]
train_bkg = trainBkgDF.loc[:,input_features]
val_sig = testSigDF.loc[:,input_features]
val_bkg = testBkgDF.loc[:,input_features]

train_sig_scaled = Scaling_train.transform(train_sig)
train_bkg_scaled = Scaling_train.transform(train_bkg)
val_sig_scaled = Scaling_train.transform(val_sig)
val_bkg_scaled = Scaling_train.transform(val_bkg)

pred_train_sig = model.predict(train_sig_scaled)
pred_train_bkg = model.predict(train_bkg_scaled)
pred_val_sig = model.predict(val_sig_scaled)
pred_val_bkg = model.predict(val_bkg_scaled)

print("plotting nn prediction")
fig = plt.figure(figsize=(15,12))
plt.style.use(hep.style.ROOT)
hep.atlas.text(text='Internal', loc=1, fontsize=20)
hep.atlas.text(text=r'$\sqrt{s}$=13 TeV, ttHH fullHad signal', loc=2, fontsize=20)
plt.hist(pred_train_sig, bins=50, histtype="step", density=True, linestyle='--', label="signal (training)", color="steelblue")
plt.hist(pred_val_sig, bins=50, histtype="step", density=True, label="signal (validation)", color="steelblue")
plt.hist(pred_val_bkg, bins=50, histtype="step", density=True, label="background (validation)", color="darkorange")
plt.hist(pred_train_bkg, bins=50, histtype="step", density=True, linestyle='--', label="background (training)", color="darkorange")

plt.legend()
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax*1.2)
plt.xlabel("DNN Prediction")
plt.ylabel("No. events")
plot_name = model_path +"/nn_pred.pdf"
plt.ioff()
fig.savefig(plot_name, transparent=True)
plt.close(fig)

################## DONE: shapley #####################
print("plotting shapley beeswarm")
fig = plt.figure(figsize=(15,12))
X_shap = shuffle(x_train_scaled, random_state=0)
explainer = shap.Explainer(model.predict, masker=X_shap)
explanation = explainer(X_shap[:200, :])
shap_values = shap.Explanation(
    values=explanation.values, 
    base_values=explanation.base_values, 
    data=explanation.data, 
    feature_names=input_features
)
shap.plots.beeswarm(shap_values, max_display=50)
plt.tight_layout()
plt.ioff()
fig.savefig(model_path + "/shapley_beeswarm.pdf")
plt.close(fig)

######################################################


################ Plot ROC, PRC and AUC ######################
y_pred_train = model.predict(x_train_scaled)
y_pred_valid = model.predict(x_valid_scaled)

# signal eff, false postivie, and true positive rates for roc curves
fpr_train, tpr_train, sig_eff_train = roc_curve(y_train, y_pred_train)
fpr_val, tpr_val, sig_eff_val = roc_curve(y_valid, y_pred_valid)
roc_auc_train = roc_auc_score(y_train, y_pred_train)
roc_auc_val = roc_auc_score(y_valid, y_pred_valid)

# precision recall and auc
prec_train, rec_train, _ = precision_recall_curve(y_train, y_pred_train)
prec_val, rec_val, _ =precision_recall_curve(y_valid, y_pred_valid)
pr_auc_train = auc(y_train, y_pred_train)
pr_auc_val = auc(y_valid, y_pred_valid)

# plot roc curve
print("plotting roc curve")
fig = plt.figure(figsize=(15,12))
plt.style.use(hep.style.ROOT)
hep.atlas.text(text='Internal', loc=1, fontsize=20)
hep.atlas.text(text='ttHH fullHad signal', loc=2, fontsize=20)
plt.plot(fpr_train, tpr_train, label='Training ROC curve (AUC = {:.3f})'.format(roc_auc_train))
plt.plot(fpr_val, tpr_val, label='Validation ROC curve (AUC = {:.3f})'.format(roc_auc_val))
plt.legend()
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax*1.2)
plt.xlim(0,1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plot_name = model_path +"/ROC.pdf"
plt.ioff()
fig.savefig(plot_name, transparent=True)
plt.close(fig)

# plot precision recall curve
print("plotting precision recall curve")
fig = plt.figure(figsize=(15,12))
plt.style.use(hep.style.ROOT)
hep.atlas.text(text='Internal', loc=1, fontsize=20)
hep.atlas.text(text='ttHH fullHad signal', loc=2, fontsize=20)
plt.plot(rec_train, prec_train, label='Training precision recall curve (AUC = {:.3f})'.format(pr_auc_train))
plt.plot(rec_val, prec_val, label='Validation precision recall curve (AUC = {:.3f})'.format(pr_auc_val))
plt.legend()
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax*1.2)
plt.xlim(0,1)
plt.xlabel("Recall")
plt.ylabel("Precision")
plot_name = model_path +"/Precision_recall.pdf"
plt.ioff()
fig.savefig(plot_name, transparent=True)
plt.close(fig)

# plot significance curve (this makes no actual sense without weights!!!)
print("plotting significance curve")
fig = plt.figure(figsize=(15,12))
plt.style.use(hep.style.ROOT)
hep.atlas.text(text='Internal', loc=1, fontsize=20)
hep.atlas.text(text='ttHH fullHad signal', loc=2, fontsize=20)
plt.plot(fpr_train, tpr_train / (fpr_train**0.5), label='Training significance curve') # is it meaningful to plot fpr as x variable?
plt.plot(fpr_val, tpr_val / (fpr_val**0.5), label='Validation significance curve')
plt.legend()
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax*1.2)
plt.xlim(0,1)
plt.xlabel("False Positive Rate")
plt.ylabel("Significance")
plot_name = model_path +"/Significance.pdf"
plt.ioff()
fig.savefig(plot_name, transparent=True)
plt.close(fig)

################ NOW WE TRY WRITING OUT THESE PREDICTION TO AN OUTPUT FILE ################
Save_pred(pred_train_sig,pred_train_bkg,pred_val_sig,pred_val_bkg, model_path)

