#!/bin/bash
source ./path.sh

echo "Executing command: feat-to-dim scp:data/train/data/raw_mfcc_train.1.scp-"
feat-to-dim scp:data/train/data/raw_mfcc_train.1.scp -
echo "-----------------------------------------"

echo "Executing command: feat-to-len scp:data/train/feats.scp ark, t:data/train/feats.lengths"
feat-to-len scp:data/train/feats.scp ark,t:data/train/feats.lengths
echo "-----------------------------------------"

echo "First 5 lines of data/train/feats.lengths:"
cat data/train/feats.lengths | head -n 5

