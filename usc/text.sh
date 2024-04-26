#!/bin/bash

awk -F'_' '{ map[$2] = $0 } END { for (num in map) print num, map[num] }' ./data/dev/uttids > temp_map.txt

# Replace the number in the sentences file with the corresponding utterance_id
awk 'NR==FNR { map[$1]=$0; next } { if ($1 in map) print map[$1], $0 }' temp_map.txt ~/Downloads/usc/transcriptions.txt  | sed 's/^[^ ]* //' > ./data/dev/text

gawk -i inplace '{output = ""; for(i=1; i < NF; i++) { if(i != 2) {output = output $i " ";}}  print output; }'  ./data/dev/text 
# Clean up temporary files
rm temp_map.txt

awk -F'_' '{ map[$2] = $0 } END { for (num in map) print num, map[num] }' ./data/train/uttids > temp_map.txt

# Replace the number in the sentences file with the corresponding utterance_id
awk 'NR==FNR { map[$1]=$0; next } { if ($1 in map) print map[$1], $0 }' temp_map.txt ~/Downloads/usc/transcriptions.txt | sed 's/^[^ ]* //' > ./data/train/text

gawk -i inplace '{output = ""; for(i=1; i < NF; i++) { if(i != 2) {output = output $i " ";} } print output; }'  ./data/train/text
# Clean up temporary files
rm temp_map.txt

awk -F'_' '{ map[$2] = $0 } END { for (num in map) print num, map[num] }' "./data/test/uttids" > temp_map.txt

# Replace the number in the sentences file with the corresponding utterance_id

awk 'NR==FNR { map[$1]=$0; next } { if ($1 in map) print map[$1], $0 }' temp_map.txt ~/Downloads/usc/transcriptions.txt | sed 's/^[^ ]* //' > ./data/test/text

gawk -i inplace '{output = ""; for(i=1; i < NF; i++) { if(i != 2) {output = output $i " ";} } print output; }'  ./data/test/text
# Clean up temporary files
#rm temp_map.txt
