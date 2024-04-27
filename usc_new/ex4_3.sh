#!/bin/bash
source ./path.sh

for x in train test dev; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 --cmd run.pl data/$x
    steps/compute_cmvn_stats.sh data/$x || exit 1;
done