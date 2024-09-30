import sys
import subprocess

def test_spell_checker(model_name, test_file="data/spell_test.txt", max_lines=20):
    """
    Tests a spell checker model against a set of misspelled words.

    Args:
        model_name (str): Name of the model to use for predictions.
        test_file (str): Path to the file containing test data.
        max_lines (int): Maximum number of lines to read from the test file.
    """

    with open(test_file, 'r') as f:
        for i, line in enumerate(f):  # Iterate over lines in the file
            if i >= max_lines:       # Stop after reading the maximum number of lines
                break

            words = line.strip().split() 
            if len(words) < 2:  # Ensure there's at least one misspelled word to test
                print(f"[WARNING] Line {i+1}: Insufficient words for testing ({line.strip()})")
                continue  

            correct = words[0].rstrip(':')  # The first word is the correct spelling
            misspelled_words = words[1:]     # The rest are misspelled versions to test

            for misspelled in misspelled_words:
                try:
                    # Run the prediction script to get the corrected word
                    result = subprocess.run(
                        ["./scripts/predict.sh", model_name, misspelled],
                        capture_output=True, text=True, check=True
                    )
                    predicted = result.stdout.strip()
                except subprocess.CalledProcessError as e:
                    # Handle errors from the prediction script
                    predicted = f"[ERROR: {e}]"

                # Print results in a clear, single-line format
                print(f"Input: {misspelled}, Correct: {correct}, Predicted: {predicted}") 


if __name__ == "__main__":
    if len(sys.argv) < 2:  # Check if the model name is provided as a command-line argument
        print("Usage: python3 step7.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    test_spell_checker(model_name)  # Start the spell checker test
