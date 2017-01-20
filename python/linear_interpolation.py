import math


# start symbol
START = '<s>'
# stop symbol
STOP = '</s>'
# unk symbol
UNK = '<UNK>'

# linear interpolation hyper-parameters
LAMBDA_1 = 0.2
LAMBDA_2 = 0.3
LAMBDA_3 = 0.5


def main():
    model = train('data/reuters_train.txt')

    # returns the log probability of a specified n-gram
    def get_log_prob(n_gram, model):
        # tri-gram part
        tri_numer = model[2][n_gram]
        tri_denom = model[1][tuple(n_gram[1:])]
        trigram_part = 0
        if tri_denom != 0:
            trigram_part = LAMBDA_1 * tri_numer / tri_denom

        # bi-gram part
        bi_numer = model[1][tuple(n_gram[1:])]
        bi_denom = model[0][tuple(n_gram[2:])]
        bigram_part = 0
        if bi_denom != 0:
            bigram_part = LAMBDA_2 * bi_numer / bi_denom

        # uni-gram part
        uni_numer = model[0][tuple(n_gram[2:])]
        uni_denom = sum(model[0].values()) - model[0][(START,)]
        unigram_part = 0
        if uni_denom != 0:
            unigram_part = LAMBDA_3 * uni_numer / uni_denom

        prob = trigram_part + bigram_part + unigram_part
        log_prob = math.log(prob, 2)
        return log_prob

    perplexity = eval_model('data/reuters_test.txt', model, get_log_prob)
    print(str(perplexity))


# returns a tuple of Counter objects (unigrams, bigrams, trigrams)
def train(filename):
    print('training...')

    # initializing empty Counter objects to store the n-grams
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()

    lines = list()
    with open(filename, encoding='iso-8859-1') as f:
        for line in f:
            lines.append(line)

    # creating the unigram counts
    for line in lines:
        tokens = line.split()
        tokens.insert(0, START)
        tokens.insert(0, START)
        tokens.append(STOP)
        add_n_gram_counts(1, unigrams, tokens)

    # the set of all unigrams that have a count of 1
    unks = set()
    for unigram, count in unigrams.items():
        if count == 1:
            unks.add(unigram)

    for unigram in unks:
        del unigrams[unigram]

    unigrams[(UNK,)] = len(unks)

    # creating the bigram and trigram counts
    for line in lines:
        tokens = [token if token not in unks else UNK for token in line.split()]
        tokens.insert(0, START)
        tokens.insert(0, START)
        tokens.append(STOP)
        add_n_gram_counts(2, bigrams, tokens)
        add_n_gram_counts(3, trigrams, tokens)

    print('finished training!')
    return unigrams, bigrams, trigrams


# adds the n-grams to the specified Counter from the specified tokens
def add_n_gram_counts(n, n_grams, tokens):
    for i in range(len(tokens) - (n - 1)):
        n_grams[tuple(tokens[i:i+n])] += 1

    return n_grams


# returns the perplexity of the model
def eval_model(filename, model, log_prob_func):
    print('evaluating model')

    log_prob_sum = 0
    file_word_count = 0

    with open(filename) as f:
        for line in f:
            prob, num_tokens = eval_sentence(line, model, log_prob_func)
            log_prob_sum += prob
            file_word_count += num_tokens
        f.close()

    print('finished evaluating!')
    average_log_prob = log_prob_sum / file_word_count
    perplexity = 2**(-average_log_prob)
    return perplexity


# returns log probability of a sentence and how many tokens were in the sentence
def eval_sentence(sentence, model, log_prob_func):
    tokens = [token if (token,) in model[0] else UNK for token in sentence.split()]
    num_tokens = len(tokens) + 1
    tokens.insert(0, START)
    tokens.insert(0, START)
    tokens.append(STOP)

    log_prob_sum = 0
    for i in range(len(tokens) - 2):
        n_gram = tuple(tokens[i:i+3])
        next_prob = log_prob_func(n_gram, model)
        log_prob_sum += next_prob

    return log_prob_sum, num_tokens


if __name__ == '__main__':
    main()
