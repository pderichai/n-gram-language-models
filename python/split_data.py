import random


lines = list()

with open('data/gutenberg.txt', encoding='iso-8859-1') as f:
    for line in f:
        lines.append(line)
    f.close()

indices = list(range(len(lines)))
random.shuffle(indices)

with open('data/gutenberg_dev.txt', 'w', encoding='iso-8859-1') as f:
    for i in range(int(0.1 * len(lines))):
        f.write(lines[indices[i]])
    f.close()

with open('data/gutenberg_test.txt', 'w', encoding='iso-8859-1') as f:
    for i in range(int(0.1 * len(lines)), int(0.2 * len(lines))):
        f.write(lines[indices[i]])
    f.close()

with open('data/gutenberg_train.txt', 'w', encoding='iso-8859-1') as f:
    for i in range(int(0.2 * len(lines)), len(lines)):
        f.write(lines[indices[i]])
    f.close()

'''with open('data/reuters.txt', encoding='iso-8859-1') as f:
    for line in f:
        lines.append(line)
    f.close()

indices = list(range(len(lines)))
random.shuffle(indices)

with open('data/reuters_dev.txt', 'w', encoding='iso-8859-1') as f:
    for i in range(int(0.1 * len(lines))):
        f.write(lines[indices[i]])
    f.close()

with open('data/reuters_test.txt', 'w', encoding='iso-8859-1') as f:
    for i in range(int(0.1 * len(lines)), int(0.2 * len(lines))):
        f.write(lines[indices[i]])
    f.close()

with open('data/reuters_train.txt', 'w', encoding='iso-8859-1') as f:
    for i in range(int(0.2 * len(lines)), len(lines)):
        f.write(lines[indices[i]])
    f.close()'''
