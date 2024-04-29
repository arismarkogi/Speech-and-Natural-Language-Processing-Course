#!/bin/bash

echo -e "Monophone Unigram Dev PER\n"
cat ./exp/mono_ug/decode_dev/scoring_kaldi/best_wer
echo -e "Monophone Unigram Test PER\n"
cat ./exp/mono_ug/decode_test/scoring_kaldi/best_wer
echo -e "Monophone Bigram Dev PER\n"
cat ./exp/mono_bg/decode_dev/scoring_kaldi/best_wer
echo -e "Monophone Bigram Test PER\n"
cat ./exp/mono_bg/decode_test/scoring_kaldi/best_wer
