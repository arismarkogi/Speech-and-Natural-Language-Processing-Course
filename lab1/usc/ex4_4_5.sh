#!/bin/bash
source ./path.sh

echo -e "Align the monophone ug model\n"
steps/align_si.sh data/train data/lang_test_ug exp/mono_ug exp/mono_aligned_ug
echo -e "Align the monophone bg model\n"
steps/align_si.sh data/train data/lang_test_bg exp/mono_bg exp/mono_aligned_bg
