#!/bin/bash
source ./path.sh

echo -e "Align the monophone model\n"
steps/align_si.sh data/train data/lang_test_bg exp/mono exp/mono_aligned

echo -e "Train triphone model"
steps/train_deltas.sh 2000 10000 data/train data/lang exp/mono_aligned exp/tri

echo -e "\n---------------------------------------------------------------------------\n"





echo -e "\nCreate HCLG for triphone unigram, aligned\n"
./utils/mkgraph.sh data/lang_test_ug exp/tri exp/tri/graph_unigram

echo -e "\nCreate HCLG for tirphone bigram, aligned\n"
./utils/mkgraph.sh data/lang_test_bg exp/tri exp/tri/graph_bigram

echo -e "\n---------------------------------------------------------------------------\n"


echo -e "\nDecode unigram with viterbi at data/dev, aligned\n"
./steps/decode.sh exp/tri/graph_unigram data/dev exp/tri/decode_dev_unigram

echo -e "\nDecode unigram with viterbi at data/test, aligned"
./steps/decode.sh exp/tri/graph_unigram data/test exp/tri/decode_test_unigram

echo -e "\nDecode bigram with viterbi at data/dev, aligned"
./steps/decode.sh exp/tri/graph_bigram data/dev exp/tri/decode_dev_bigram

echo -e "\nDecode bigram with viterbi at data/test, aligned"
./steps/decode.sh exp/tri/graph_bigram data/test exp/tri/decode_test_bigram