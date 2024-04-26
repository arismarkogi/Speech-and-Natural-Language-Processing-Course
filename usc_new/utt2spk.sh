#!/bin/bash

awk -F'_' '{print $0, $1}' ./data/dev/uttids > ./data/dev/utt2spk
awk -F'_' '{print $0, $1}' ./data/train/uttids > ./data/train/utt2spk
awk -F'_' '{print $0, $1}' ./data/test/uttids > ./data/test/utt2spk
