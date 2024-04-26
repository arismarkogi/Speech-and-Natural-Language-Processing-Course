import re

def main():
    files = ['/home/arismarkogi/kaldi/egs/usc/data/train/text','/home/arismarkogi/kaldi/egs/usc/data/test/text', '/home/arismarkogi/kaldi/egs/usc/data/dev/text']
    try:
        lexicon = {}  # Assuming lexicon is a dictionary
        with open('/home/arismarkogi/Downloads/usc/lexicon.txt', 'r+') as file:
            for line in file:
                if '\t' in line:
                    first_word, rest_of_sentence = line.split('\t')
                else:
                    first_word, rest_of_sentence = line.split(' ', 1)
        
                rest_of_sentence = rest_of_sentence.rstrip('\n')  # Remove newline character
        
                lexicon[first_word.lower()] = rest_of_sentence

        for i in range(3):
            print(files[i])
            with open(files[i], 'r+') as file:
                lines = file.readlines()
                file.seek(0)  # Move the file pointer to the beginning
                for line in lines:
                    word_separator_index = line.find(' ')
                    if word_separator_index != -1:  # Ensure there's at least one space
                        first_word = line[:word_separator_index] 
                        sentence = line[word_separator_index+1:].strip()  # Extract the sentence
                        words = sentence.split(' ')
                        phonemes = []
                        for word in words:
                            word = word.lower()
                            word = re.sub(r"[^\w\s']", '', word)
                            phonemes.append(lexicon.get(word, ''))
                
                        phonemes = " ".join(phonemes)
                        text_to_write = first_word + " sil" + phonemes + " sil\n"  # Add newline character
            
                        file.write(text_to_write)  # Write the new text to the file
            
                file.truncate()  # Truncate the remaining content if any (if new text is shorter)
                
            print("Text has been successfully copied from '{}' to '{}'.".format(files[i], files[i]))
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
