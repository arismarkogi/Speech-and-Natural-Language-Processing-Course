#!/bin/bash
source ./path.sh
cd "data/local/lm_tmp"

echo "# Calculating perplexity for the bigram model of the dev set #"
compile-lm lm_output_trainb.ilm.gz --eval=../dict/lm_dev.txt --dub=10000000

echo "# Calculating perplexity for the unigram model of the dev set #"
compile-lm lm_output_trainu.ilm.gz --eval=../dict/lm_dev.txt --dub=10000000

echo "# Calculating perplexity for the bigram model of the test set #"
compile-lm lm_output_trainb.ilm.gz --eval=../dict/lm_test.txt --dub=10000000

echo "# Calculating perplexity for the unigram model of the test set #"
compile-lm lm_output_trainu.ilm.gz --eval=../dict/lm_test.txt --dub=10000000
cd ../../..