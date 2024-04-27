#!/bin/bash
source ./path.sh

echo -e "Align the monophone model\n"
steps/align_si.sh data/train data/lang_test_bg exp/mono exp/mono_aligned

echo -e "\n---------------------------------------------------------------------------\n"


echo -e "\nCreate HCLG for unigram, aligned\n"
./utils/mkgraph.sh data/lang_test_ug exp/mono_aligned exp/mono_aligned/graph_unigram

echo -e "\nCreate HCLG for bigram, aligned\n"
./utils/mkgraph.sh data/lang_test_bg exp/mono_aligned exp/mono_aligned/graph_bigram

echo -e "\n---------------------------------------------------------------------------\n"


echo -e "\nDecode unigram with viterbi at data/dev, aligned\n"
./steps/decode.sh exp/mono_aligned/graph_unigram data/dev exp/mono_aligned/decode_dev_unigram

echo -e "\nDecode unigram with viterbi at data/test, aligned"
./steps/decode.sh exp/mono_aligned/graph_unigram data/test exp/mono_aligned/decode_test_unigram

echo -e "\nDecode bigram with viterbi at data/dev, aligned"
./steps/decode.sh exp/mono_aligned/graph_bigram data/dev exp/mono_aligned/decode_dev_bigram

echo -e "\nDecode bigram with viterbi at data/test, aligned"
./steps/decode.sh exp/mono_aligned/graph_bigram data/test exp/mono_aligned/decode_test_bigram