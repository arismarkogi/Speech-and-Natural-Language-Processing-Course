#!/bin/bash

source ./path.sh

mkdir data/local/lm_tmp

build-lm.sh -i data/local/dict/lm_train.txt -n 1 -o data/local/lm_tmp/lm_output_trainu.ilm.gz
build-lm.sh -i data/local/dict/lm_train.txt -n 2 -o data/local/lm_tmp/lm_output_trainb.ilm.gz

build-lm.sh -i data/local/dict/lm_test.txt -n 1 -o data/local/lm_tmp/lm_output_testu.ilm.gz
build-lm.sh -i data/local/dict/lm_test.txt -n 2 -o data/local/lm_tmp/lm_output_testb.ilm.gz


build-lm.sh -i data/local/dict/lm_dev.txt -n 1 -o data/local/lm_tmp/lm_output_devu.ilm.gz
build-lm.sh -i data/local/dict/lm_dev.txt -n 2 -o data/local/lm_tmp/lm_output_devb.ilm.gz