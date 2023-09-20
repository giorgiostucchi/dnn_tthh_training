Based on dnn_training from HHH->6b analysis https://gitlab.cern.ch/trihiggs/dnn_training 

# DNN_training
The modules needed in your environment for the training are specified in requirements.txt.

## Training
The DNN training for resonant TRSM uses
``` 
train.py --LR <learning-rate>' --batchsize <batchsize> --node <node> --layer <layer> --dropout <rate>
```
where the learning rate, batchsize, the number of nodes per layer, the number of hidden layers, dropout rate, and even/odd fold needs to be specified. The training files, and output file path where the model and model history are saved need to be specified inside (should probably make this configurable at some point.)
The SM HHH DNN training uses 
``` 
train_SM.py --LR <learning-rate>' --batchsize <batchsize> --node <node> --layer <layer> --dropout <rate>
```
Same as above. The input features used in the resonant TRSM and SM HHH training are different, and are optimised separately.

## Hyperparameter scan
The hyperparameters of these models can be optimised by running a grid search by running
```
./parameter_scan.py
```
where the parameter grid, and model can be specified inside. A small number of epochs (e.g. 10 or 20) is sufficient per model.
After the scan, run
```
python parameter_scan.py
```
to get the parameters of the model with the maximum ROC AUC evaluated on the validation dataset.
