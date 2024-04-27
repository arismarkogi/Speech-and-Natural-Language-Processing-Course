#!/bin/bash
source ./path.sh

echo "Train monophone model"
./steps/train_mono.sh data/train data/lang exp/mono


