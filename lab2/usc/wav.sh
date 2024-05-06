#!/bin/bash

awk -v path="~/Downloads.usc/wav/" -F' ' '{print $1, path $1 ".wav"}' ./data/dev/utt2spk > ./data/dev/wav.scp
awk -v path="~/Downloads.usc/wav/" -F' ' '{print $1, path $1 ".wav"}' ./data/train/utt2spk > ./data/train/wav.scp
awk -v path="~/Downloads.usc/wav/" -F' ' '{print $1, path $1 ".wav"}' ./data/test/utt2spk > ./data/test/wav.scp
