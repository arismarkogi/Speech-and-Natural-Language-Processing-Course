#!/bin/bash
source ./path.sh

compile-lm data/local/lm_tmp/lm_output_devu.ilm.gz --eval=data/local/dict/lm_dev.txt --dub=10000000

compile-lm data/local/lm_tmp/lm_output_devb.ilm.gz --eval=data/local/dict/lm_dev.txt --dub=10000000

compile-lm data/local/lm_tmp/lm_output_testu.ilm.gz --eval=data/local/dict/lm_test.txt --dub=10000000

compile-lm data/local/lm_tmp/lm_output_testb.ilm.gz --eval=data/local/dict/lm_test.txt --dub=10000000

