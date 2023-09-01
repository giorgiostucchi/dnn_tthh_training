#!/bin/bash

LR=(0.005 0.01)
BATCHSIZE=(128 256)
NODE=(12 24)
LAYER=(3 4)
DROPOUT=(0.1 0.2)

for lr in ${LR[@]}
do
    for bs in ${BATCHSIZE[@]}
    do 
        for node in ${NODE[@]}
        do 
            for layer in ${LAYER[@]}
            do
                for dropout in ${DROPOUT[@]}
                do
                    echo "LR:"$lr" Batchsize:"$bs" Node:"$node" Layer:"$layer" Dropout:"$dropout
                    python train_SM.py --LR $lr --batchsize $bs --node $node --layer $layer --dropout $dropout
                    done
            done
        done
    done
done