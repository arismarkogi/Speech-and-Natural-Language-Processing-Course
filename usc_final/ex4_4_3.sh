#!/bin/bash
source ./path.sh

echo "Decode unigram with viterbi at data/dev"
./steps/decode.sh exp/mono_ug/graph data/dev exp/mono_ug/decode_dev

echo -e "\nDecode unigram with viterbi at data/test"
./steps/decode.sh exp/mono_ug/graph data/test exp/mono_ug/decode_test

echo -e "\nDecode bigram with viterbi at data/dev"
./steps/decode.sh exp/mono_bg/graph data/dev exp/mono_bg/decode_dev

echo -e "\nDecode bigram with viterbi at data/test"
./steps/decode.sh exp/mono_bg/graph data/test exp/mono_bg/decode_test
