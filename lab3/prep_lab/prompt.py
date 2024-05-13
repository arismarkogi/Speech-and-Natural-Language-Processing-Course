
import sys
import random

random.seed(42)
print("Can you classify these sentences as positive or negative?()")
with open('datasets/MR/rt-polarity.neg') as f:
    lines = f.readlines()
    random_lines = random.sample(lines, 20)
    for i, line in enumerate(random_lines):
        print(f"{i+1}.",  line.strip())
with open('datasets/MR/rt-polarity.pos') as f:
    lines = f.readlines()
    random_lines = random.sample(lines, 20)
    for i, line in enumerate(random_lines):
        print(f"{i+21}.",  line.strip())

print("Can you classify these sentences as positive, negative or neutral?")

with open('datasets/Semeval2017A/gold/SemEval2017-task4-test.subtask-A.english.txt') as f:
    lines = f.readlines()
    positive_lines = []
    negative_lines = []
    neutral_lines = []
    for line in lines:
        values = line.strip().split('\t')
        if values[1] == 'positive':
            positive_lines.append(values[2])
        elif values[1] == 'negative':
            negative_lines.append(values[2])
        else:
            neutral_lines.append(values[2])
    random_positive_lines = random.sample(positive_lines, 20)
    random_negative_lines = random.sample(negative_lines, 20)
    random_neutral_lines = random.sample(neutral_lines, 20)
    for i, line in enumerate(random_positive_lines):
        print(f"{i+1}.",  line.strip())
    for i, line in enumerate(random_negative_lines):
        print(f"{i+21}.",  line.strip())
    for i, line in enumerate(random_neutral_lines):
        print(f"{i+41}.",  line.strip())