#!/bin/bash
source ./path.sh

echo "Create HCLG for unigram"
./utils/mkgraph.sh data/lang_test_ug exp/mono_ug exp/mono_ug/graph

echo "----------------------------"


echo "Create HCLG for bigram"
./utils/mkgraph.sh data/lang_test_bg exp/mono_bg exp/mono_bg/graph
