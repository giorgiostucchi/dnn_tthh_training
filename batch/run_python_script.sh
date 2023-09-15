#!/bin/bash

# for tracking when and where you run
hostname
date

LR=$1
BATCHSIZE=$2
NODE=$3
LAYER=$4
DROPOUT=$5

# Rest of your script
echo "LR: $LR Batchsize: $BATCHSIZE Node: $NODE Layer: $LAYER Dropout: $DROPOUT"

tmpDir=`mktemp -d`
cd ${tmpDir}
echo 'we are in '${PWD}

shopt -s expand_aliases
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
alias setupATLAS='source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh'
setupATLAS -q
asetup Athena,master,latest,here

cp /afs/cern.ch/user/g/gstucchi/dnn_tthh_training/train_SM.py ./

# Run your Python script with the provided arguments
echo "Running Python script..."
python train_SM.py --LR $LR --batchsize $BATCHSIZE --node $NODE --layer $LAYER --dropout $DROPOUT


cd -
echo 'we are back in '${PWD}
rm -rf ${tmpDir}