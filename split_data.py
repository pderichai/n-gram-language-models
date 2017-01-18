import random


lines = list()

with open('data/brown.txt') as f:
    for line in f:
        lines.append(line)
    f.close()

indices = list(range(len(lines)))
random.shuffle(indices)

with open('data/brown_dev.txt', 'w') as f:
    for i in range(int(0.1 * len(lines))):
        f.write(lines[indices[i]])
    f.close()

with open('data/brown_test.txt', 'w') as f:
    for i in range(int(0.1 * len(lines)), int(0.2 * len(lines))):
        f.write(lines[indices[i]])
    f.close()

with open('data/brown_train.txt', 'w') as f:
    for i in range(int(0.2 * len(lines)), len(lines)):
        f.write(lines[indices[i]])
    f.close()
