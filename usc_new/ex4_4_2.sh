#!/bin/bash
source ./path.sh

echo "Create HCLG for unigram"
./utils/mkgraph.sh data/lang_test_ug exp/mono exp/mono/graph_unigram

echo "----------------------------"


echo "Create HCLG for bigram"
./utils/mkgraph.sh data/lang_test_bg exp/mono exp/mono/graph_bigram