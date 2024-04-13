#!/bin/bash

compile-lm ../lm_tmp/lm_outputu.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > lm_phone_ug.arpa.gz
compile-lm ../lm_tmp/lm_outputb.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > lm_phone_bg.arpa.gz
