import argparse
#from ROOT import *
import pandas as pd
import numpy 
import uproot
import os
import random
import seaborn as sn
import mplhep as hep
import argparse
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from tensorflow import keras
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
from sklearn_export import Export
from sklearn.model_selection import train_test_split
import json
import shap
import h5py
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

def GetParser():
  """Argument parser for reading Ntuples script."""
  parser = argparse.ArgumentParser(
      description="Reading Ntuples command line options."
  )

  parser.add_argument(
      "--which",
      type=str,
      help="specify training even or odd model",
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
input_features = ["rmsmBB","rmsdABB","aplan4dv2b","massfraceta","htbjet","mHcosTheta","sphere3dbtrans","bjetH1_dR","bjetH2_dR","bjetH3_dR"]

DEFAULT = -999
LR = args.LR
BATCHSIZE = args.batchsize
EPOCH=70
nodes = [args.node]*args.layer
dropout_rate = args.dropout

# Function for saving keras predictions
def Save_pred(pred_train_sig,pred_train_bkg,pred_test_sig,pred_test_bkg, path):
  numpy.savetxt(str(path) + "/TrainSigPred.txt", pred_train_sig, delimiter=',')
  numpy.savetxt(str(path) + "/TrainBkgPred.txt", pred_train_bkg, delimiter=',')
  numpy.savetxt(str(path) + "/TestSigPred.txt", pred_test_sig, delimiter=',')
  numpy.savetxt(str(path) + "/TestBkgPred.txt", pred_test_bkg, delimiter=',')

######### classification model ###################
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
even_bkg_file = uproot.open("/data/atlas/atlasdata3/maggiechen/HHH6b_pairing/pairing+nn/20230501_pTreco_off/mH-120_115_110_pairing/training_inclusive/5b_data_even.root:tree")
odd_bkg_file = uproot.open("/data/atlas/atlasdata3/maggiechen/HHH6b_pairing/pairing+nn/20230501_pTreco_off/mH-120_115_110_pairing/training_inclusive/5b_data_odd.root:tree")
df_bkg_even = even_bkg_file.arrays(library="pd")
df_bkg_odd = odd_bkg_file.arrays(library="pd")

columns = even_bkg_file.keys()
trainSigDF = pd.DataFrame(columns = columns)
trainBkgDF = pd.DataFrame(columns = columns)
testSigDF = pd.DataFrame(columns = columns)
testBkgDF = pd.DataFrame(columns = columns)
trainDF = pd.DataFrame(columns = columns)
testDF = pd.DataFrame(columns = columns)
train_sig_sample_wgts = []

num_train_sig = 0
num_test_sig = 0
DSID = [521176, 521177, 521178, 521179, 521180, 521181, 521182, 521183, 521184, 521185, 521186, 521187, 521188]
for i, id in enumerate(DSID):
  print("reading in signal files", id)
  sig_even_file = uproot.open("/data/atlas/atlasdata3/maggiechen/HHH6b_pairing/pairing+nn/20230501_pTreco_off/mH-120_115_110_pairing/training_inclusive/6b_res_TRSM_"+str(id)+"_even.root:tree")
  sig_odd_file = uproot.open("/data/atlas/atlasdata3/maggiechen/HHH6b_pairing/pairing+nn/20230501_pTreco_off/mH-120_115_110_pairing/training_inclusive/6b_res_TRSM_"+str(id)+"_odd.root:tree")
  df_sig_even = sig_even_file.arrays(library="pd")
  df_sig_odd = sig_odd_file.arrays(library="pd")

  if args.which=="even":
    trainSigDF = trainSigDF.append(df_sig_even, ignore_index=True)
    testSigDF = testSigDF.append(df_sig_odd, ignore_index=True)
    train_sig_wgts = df_sig_even.loc[:, "eventWeight"].values
    len_train_seg = len(df_sig_even)
    len_test_seg = len(df_sig_odd)
  elif args.which == "odd":
    trainSigDF = trainSigDF.append(df_sig_odd, ignore_index=True)
    testSigDF = testSigDF.append(df_sig_even, ignore_index=True)
    train_sig_wgts = df_sig_odd.loc[:, "eventWeight"].values
    len_train_seg = len(df_sig_odd)
    len_test_seg = len(df_sig_even)
  else:
    print("choose to train either even or odd model!!")
  
  num_train_sig += len_train_seg
  num_test_sig += len_test_seg
  # weight event weights by number of events in each mass point
  train_sig_sample_wgts.append(train_sig_wgts*len_train_seg)

# then dividing the event weights by number of total signal events
train_sig_sample_wgts = numpy.concatenate(train_sig_sample_wgts).ravel() / num_train_sig

# scaling the signal sample weights to have mean = 1 before adding in background sample weights
mean_sig_train_wgts = numpy.mean(train_sig_sample_wgts)
train_sig_sample_wgts += (1-mean_sig_train_wgts)
trainSigDF["sampleWeight"] = train_sig_sample_wgts

train_sig_y = [1]*num_train_sig
test_sig_y = [1]*num_test_sig

# split training/validation signal events
x_sig_train, x_sig_valid, y_sig_train, y_sig_valid = train_test_split(trainSigDF, train_sig_y, test_size=0.3, shuffle= True)

# Add background events and sample weights to training / testing sample
if args.which == "even":
  trainBkgDF = trainBkgDF.append(df_bkg_even, ignore_index=True)
  testBkgDF = testBkgDF.append(df_bkg_odd, ignore_index=True)
  train_bkg_sample_wgts = df_bkg_even.loc[:, "eventWeight"].values
elif args.which == "odd":
  trainBkgDF = trainBkgDF.append(df_bkg_odd, ignore_index=True)
  testBkgDF = testBkgDF.append(df_bkg_even, ignore_index=True)
  train_bkg_sample_wgts = df_bkg_odd.loc[:, "eventWeight"].values
else:
  print("choose to train either even or odd model!!")
trainBkgDF["sampleWeight"] = train_bkg_sample_wgts

# backgorund event labels
train_bkg_y = [0]*len(trainBkgDF)
test_bkg_y = [0]*len(testBkgDF)

# split training/validation background events
x_bkg_train, x_bkg_valid, y_bkg_train, y_bkg_valid = train_test_split(trainBkgDF, train_bkg_y, test_size=0.3, shuffle= True)

class_weight = {0: 1.,
                1: len(x_bkg_train)/len(x_sig_train)}

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
print("Signal yields in training", sum(x_sig_train["sampleWeight"]))

testDF = testDF.append(testSigDF, ignore_index=True)
testDF = testDF.append(testBkgDF, ignore_index=True)
y_test = numpy.append([1]*len(testSigDF), [0]*len(testBkgDF))
x_test = testDF.loc[:, input_features]

print("Training variables:", input_features)
nFeatures = len(input_features)

model_path = "20230501_pTreco_off/optimisation/baseline_pairing_nobtag_opt_plots_"+args.which
os.makedirs(model_path, exist_ok=True)
# Saving training variables to json
var_dict = {}
variables = trainDF.columns.tolist()
var_dict["variables"] = variables
with open(model_path + "/variables.json", "w") as var_file:
  json.dump(var_dict, var_file)

########### Standardising training and validation data ###########
scaler = StandardScaler()
Scaling_train = scaler.fit(x_train)
x_train_scaled = Scaling_train.transform(x_train)
x_valid_scaled = Scaling_train.transform(x_valid)
x_test_scaled = Scaling_train.transform(x_test)
scaling = Export(scaler)
scale_dict = scaling.to_json()
with open(model_path + "/scaling.json", "w") as outfile:
  json.dump(scale_dict, outfile)


# saving training/validation/test datasets to h5
file_path = "/data/atlas/atlasdata3/maggiechen/HHH6b_pairing/pairing+nn/20230501_pTreco_off/mH-120_115_110_pairing/training_inclusive/h5py/"
train_file = file_path + str(args.which) + "_model_scaled_train.h5"
valid_file = file_path + str(args.which) + "_model_scaled_valid.h5"
test_file = file_path + str(args.which) + "_model_scaled_test.h5"

with h5py.File(train_file, 'w') as train_f:
  train_f.create_dataset('inputs', data=x_train_scaled)
  train_f.create_dataset('labels', data=y_train)

with h5py.File(valid_file, 'w') as valid_f:
  valid_f.create_dataset('inputs', data=x_valid_scaled)
  valid_f.create_dataset('labels', data=y_valid)

with h5py.File(test_file, 'w') as test_f:
  test_f.create_dataset('inputs', data=x_test_scaled)
  test_f.create_dataset('labels', data=y_test)


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
fig = plt.figure(figsize=(15,12))
plt.style.use(hep.style.ROOT)
hep.atlas.text(text='Internal', loc=1, fontsize=20)
hep.atlas.text(text=r'$\sqrt{s}$=13 TeV, 5b Data'+'\n'+'6b Resonant TRSM signal', loc=2, fontsize=20)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'validation'], loc='upper right')
plt.ylabel('Loss')
plt.xlabel('Epoch')
name = model_path + "/TrainValLoss.pdf"
fig.savefig(name)

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

fig = plt.figure(figsize=(15,12))
plt.style.use(hep.style.ROOT)
hep.atlas.text(text='Internal', loc=1, fontsize=20)
hep.atlas.text(text=r'$\sqrt{s}$=13 TeV, 5b Data'+'\n'+'6b Resonant TRSM signal', loc=2, fontsize=20)
plt.hist(pred_train_sig, bins=50, histtype="step", density=True, linestyle='--', label="6b resonant TRSM (training)", color="steelblue")
plt.hist(pred_val_sig, bins=50, histtype="step", density=True, label="6b resonant TRSM (validation)", color="steelblue")
plt.hist(pred_train_bkg, bins=50, histtype="step", density=True, linestyle='--', label="5b data (training)", color="darkorange")
plt.hist(pred_val_bkg, bins=50, histtype="step", density=True, label="5b data (validation)", color="darkorange")
plt.legend()
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax*1.2)
plt.xlabel("DNN Prediction")
plt.ylabel("No. events")
plot_name = model_path +"/nn_pred.pdf"
fig.savefig(plot_name, transparent=True)

################## DONE: shapley #####################
X_shap = shuffle(x_train_scaled, random_state=0)
explainer = shap.Explainer(model.predict, masker=X_shap)
explanation = explainer(X_shap[:200, :])
shap_values = shap.Explanation(
    values=explanation.values, 
    base_values=explanation.base_values, 
    data=explanation.data, 
    feature_names=input_features
)
import matplotlib.pyplot as plt
["rmsmBB","rmsdABB","aplan4dv2b","massfraceta","htbjet","mHcosTheta","sphere3dbtrans","bjetH1_dR","bjetH2_dR","bjetH3_dR"]
custom_feature_names = ["RMS $m_H$","RMS $\\Delta A_H$","Aplanarity$_{6 jets}$","$\eta - m_{HHH}$ fraction","HT$_{6 jets}$","$m_H cos\\theta$","Transverse Sphericity$_{6 jets}$","$\\Delta$ R$_{H1}$","$\\Delta$ R$_{H2}$","$\\Delta$ R$_{H3}$"]
shap_values.feature_names = custom_feature_names
plt.figure()
shap.plots.beeswarm(shap_values,
                    max_display=50)
plt.tight_layout()
plt.savefig(model_path + "/shapley_beeswarm.pdf")
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
fig = plt.figure(figsize=(15,12))
plt.style.use(hep.style.ROOT)
hep.atlas.text(text='Internal', loc=1, fontsize=20)
hep.atlas.text(text='6b resonant TRSM signal MC, 5b Data', loc=2, fontsize=20)
plt.plot(fpr_train, tpr_train, label='Training ROC curve (AUC = {:.3f})'.format(roc_auc_train))
plt.plot(fpr_val, tpr_val, label='Validation ROC curve (AUC = {:.3f})'.format(roc_auc_val))
plt.legend()
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax*1.2)
plt.xlim(0,1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plot_name = model_path +"/ROC.pdf"
fig.savefig(plot_name, transparent=True)

# plot precision recall curve
fig = plt.figure(figsize=(15,12))
plt.style.use(hep.style.ROOT)
hep.atlas.text(text='Internal', loc=1, fontsize=20)
hep.atlas.text(text='6b resonant TRSM signal MC, 5b Data', loc=2, fontsize=20)
plt.plot(rec_train, prec_train, label='Training precision recall curve (AUC = {:.3f})'.format(pr_auc_train))
plt.plot(rec_val, prec_val, label='Validation precision recall curve (AUC = {:.3f})'.format(pr_auc_val))
plt.legend()
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax*1.2)
plt.xlim(0,1)
plt.xlabel("Recall")
plt.ylabel("Precision")
plot_name = model_path +"/Precision_recall.pdf"
fig.savefig(plot_name, transparent=True)

################ NOW WE TRY WRITING OUT THESE PREDICTION TO AN OUTPUT FILE ################
Save_pred(pred_train_sig,pred_train_bkg,pred_val_sig,pred_val_bkg, model_path)
