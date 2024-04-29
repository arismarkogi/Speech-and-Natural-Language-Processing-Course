#!/bin/bash

source ./path.sh

compile-lm data/local/lm_tmp/lm_output_trainu.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_train_ug.arpa.gz
compile-lm data/local/lm_tmp/lm_output_trainb.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_train_bg.arpa.gz

compile-lm data/local/lm_tmp/lm_output_testu.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_test_ug.arpa.gz
compile-lm data/local/lm_tmp/lm_output_testb.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_test_bg.arpa.gz

compile-lm data/local/lm_tmp/lm_output_devu.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_dev_ug.arpa.gz
compile-lm data/local/lm_tmp/lm_output_devb.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone__dev_bg.arpa.gz