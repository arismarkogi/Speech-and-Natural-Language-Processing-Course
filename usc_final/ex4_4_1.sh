#!/bin/bash
source ./path.sh

echo "Train monophone model"
./steps/train_mono.sh data/train data/lang_test_bg exp/mono_bg

./steps/train_mono.sh data/train data/lang_test_ug exp/mono_ug

