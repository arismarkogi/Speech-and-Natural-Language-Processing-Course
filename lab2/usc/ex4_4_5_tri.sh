#!/bin/bash

source ./path.sh

echo -e "Train triphone models"
steps/train_deltas.sh 2000 10000 data/train data/lang_test_ug exp/mono_aligned_ug exp/tri_ug
steps/train_deltas.sh 2000 10000 data/train data/lang_test_bg exp/mono_aligned_bg exp/tri_bg

echo -e "\n---------------------------------------------------------------------------\n"

echo -e "\nCreate HCLG for triphone unigram, aligned\n"
./utils/mkgraph.sh data/lang_test_ug exp/tri_ug exp/tri_ug/graph

echo -e "\nCreate HCLG for tirphone bigram, aligned\n"
./utils/mkgraph.sh data/lang_test_bg exp/tri_bg exp/tri_bg/graph

echo -e "\n---------------------------------------------------------------------------\n"


echo -e "\nDecode unigram with viterbi at data/dev, aligned\n"
./steps/decode.sh exp/tri_ug/graph data/dev exp/tri_ug/decode_dev

echo -e "\nDecode unigram with viterbi at data/test, aligned"
./steps/decode.sh exp/tri_ug/graph data/test exp/tri_ug/decode_test

echo -e "\nDecode bigram with viterbi at data/dev, aligned"
./steps/decode.sh exp/tri_bg/graph data/dev exp/tri_bg/decode_dev

echo -e "\nDecode bigram with viterbi at data/test, aligned"
./steps/decode.sh exp/tri_bg/graph data/test exp/tri_bg/decode_test

echo -e "Triphone Unigram Dev PER\n"
cat ./exp/tri_ug/decode_dev/scoring_kaldi/best_wer
echo -e "Triphone Unigram Test PER\n"
cat ./exp/tri_ug/decode_test/scoring_kaldi/best_wer
echo -e "Triphone Bigram Dev PER\n"
cat ./exp/tri_bg/decode_dev/scoring_kaldi/best_wer
echo "Triphone Bigram Test PER\n"
cat ./exp/tri_bg/decode_test/scoring_kaldi/best_wer
