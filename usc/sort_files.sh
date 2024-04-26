sort -u -o data/train/wav.scp data/train/wav.scp
sort -u -o data/train/text data/train/text
sort -u -o data/train/utt2spk data/train/utt2spk

sort -u -o data/dev/wav.scp data/dev/wav.scp
sort -u -o data/dev/text data/dev/text
sort -u -o data/dev/utt2spk data/dev/utt2spk

sort -u -o data/test/wav.scp data/test/wav.scp
sort -u -o data/test/text data/test/text
sort -u -o data/test/utt2spk data/test/utt2spk
