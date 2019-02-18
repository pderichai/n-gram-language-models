START = '<S>'
'''Start symbol---two are prepended to the start of every sentence.'''

STOP = '</S>'
'''Stop symbol---one is appended to the end of every sentence.'''

UNK = '<UNK>'
'''Unknown word symbol used to describe any word that is out of vocabulary.'''

UNK_THRESHOLD = 2
'''If the count of a token is less than this number, it should be treated as
out of vocabulary.'''

TRAIN = 'data/brown-train.txt'
DEV = 'data/brown-dev.txt'
TEST = 'data/brown-test.txt'

# linear interpolation hyper-parameters
LAMBDA_1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
LAMBDA_2s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
LAMBDA_1 = 0.3
LAMBDA_2 = 0.5
LAMBDA_3 = 0.2

# add-K smoothing hyper-parameters
K = 0.001

# combined smoothing hyper-parameters
LAMBDA_1_COMBINED = 0.3
LAMBDA_2_COMBINED = 0.5
LAMBDA_3_COMBINED = 0.2
K_COMBINED = 1e-5
