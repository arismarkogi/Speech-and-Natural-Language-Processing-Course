#!/bin/bash
source ./path.sh

echo "Decode unigram with viterbi at data/dev"
./steps/decode.sh exp/mono/graph_unigram data/dev exp/mono/decode_dev_unigram

echo -e "\nDecode unigram with viterbi at data/test"
./steps/decode.sh exp/mono/graph_unigram data/test exp/mono/decode_test_unigram

echo -e "\nDecode bigram with viterbi at data/dev"
./steps/decode.sh exp/mono/graph_bigram data/dev exp/mono/decode_dev_bigram

echo -e "\nDecode bigram with viterbi at data/test"
./steps/decode.sh exp/mono/graph_bigram data/test exp/mono/decode_test_bigram