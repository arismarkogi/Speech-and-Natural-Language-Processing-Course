#!/bin/bash

cut -d ' ' -f 2- ~/Downloads/usc/lexicon.txt |  sed 's/ /\n/g' | sort -u > nonsilence_phones.txt
