#!/bin/bash
source ./path.sh

echo -e "Align the triphone bg model\n"

steps/align_si.sh data/train data/lang_test_bg exp/tri_bg exp/tri_aligned_train
steps/align_si.sh data/dev data/lang_test_bg exp/tri_bg exp/tri_aligned_dev
steps/align_si.sh data/test data/lang_test_bg exp/tri_bg exp/tri_aligned_test
