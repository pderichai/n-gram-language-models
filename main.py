import math
from collections import Counter


# start symbol
START = '<s>'
# stop symbol
STOP = '</s>'
# unk symbol
UNK = '<UNK>'
# katz-backoff discount hyper-parameter
DISCOUNT = 0.5


def main():
    unigrams, bigrams, trigrams = train('data/brown_train.txt')
    perplexity = eval_model('data/brown_dev.txt', unigrams, bigrams, trigrams)
    '''sentence = 'The marines let him advance .'
    sentence_perplexity = eval_sentence(sentence, unigrams, bigrams, trigrams) / len(sentence.split())
    print(str(sentence_perplexity))
    #print(str(get_prob(NGram(('Thus', ',', 'while')), unigrams, bigrams, trigrams, dict(), dict())))
    #print(str(get_backoff_denom(NGram((',',)), unigrams, bigrams, trigrams, dict(), dict())))'''


# returns a tuple of Counter objects (unigrams, bigrams, trigrams)
def train(filename):
    print('training...')

    # initializing empty Counter objects to store the n-grams
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()

    lines = list()
    with open(filename) as f:
        for line in f:
            lines.append(line)

    # creating the unigram counts
    for line in lines:
        tokens = line.split()
        tokens.insert(0, START)
        tokens.insert(0, START)
        tokens.append(STOP)
        get_n_gram_counts(1, unigrams, tokens)

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
        get_n_gram_counts(2, bigrams, tokens)
        get_n_gram_counts(3, trigrams, tokens)

    print('finished training!')
    return unigrams, bigrams, trigrams


# helper method to add the n-gram counts of the specified tokens to the n_grams Counter object
def get_n_gram_counts(n, n_grams, tokens):
    for i in range(len(tokens) - (n - 1)):
        n_grams[tuple(tokens[i:i+n])] += 1

    return n_grams


# returns the perplexity of the model
def eval_model(filename, unigrams, bigrams, trigrams):
    print('evaluating model')

    with open(filename) as f:
        overall_prob = 1

        for line in f:
            print('getting a prob for the line:')
            print('\t' + line)
            prob = eval_sentence(line, unigrams, bigrams, trigrams)
            print('got a prob ' + str(prob))
            overall_prob *= prob

        f.close()
        print('finished evaluating!')
        return overall_prob


# returns log probability of a sentence
def eval_sentence(sentence, unigrams, bigrams, trigrams):
    tokens = [token if (token,) in unigrams.keys() else UNK for token in sentence.split()]
    tokens.insert(0, START)
    tokens.insert(0, START)
    tokens.append(STOP)

    prob = 1
    history_to_alphas = dict()
    history_to_denoms = dict()
    n_grams_to_probs = dict()

    for i in range(len(tokens) - 2):
        n_gram = tuple(tokens[i:i+3])
        print('getting the probability for the n_gram ' + str(n_gram))
        next_prob = get_prob(n_gram, unigrams, bigrams, trigrams, n_grams_to_probs, history_to_alphas, history_to_denoms)
        print('got the probability: ' + str(next_prob))
        prob *= next_prob

    log_prob = math.log(prob, 2)
    return log_prob


# returns the probability of a specified n-gram in the model
def get_prob(n_gram, unigrams, bigrams, trigrams, n_grams_to_probs, history_to_alphas, history_to_denoms):
    if n_gram in n_grams_to_probs:
        return n_grams_to_probs[n_gram]

    if len(n_gram) == 0:
        return 0

    if n_gram_seen_in_training(n_gram, unigrams, bigrams, trigrams) or len(n_gram) == 1:
        return get_discounted_MLE_prob(n_gram, unigrams, bigrams, trigrams)

    next_gram = tuple(n_gram[1:])
    history_n_gram = tuple(n_gram[:-1])
    numer = get_prob(next_gram, unigrams, bigrams, trigrams, n_grams_to_probs, history_to_alphas, history_to_denoms)
    denom = get_backoff_denom(history_n_gram, unigrams, bigrams, trigrams, n_grams_to_probs, history_to_alphas, history_to_denoms)
    alpha = get_alpha(history_n_gram, unigrams, bigrams, trigrams, history_to_alphas)

    prob = alpha * numer / denom
    n_grams_to_probs[n_gram] = prob

    return prob


# given a specified n-gram, returns whether or not it was seen in the training data
def n_gram_seen_in_training(n_gram, unigrams, bigrams, trigrams):
    if len(n_gram) == 1:
        return n_gram in unigrams.keys()

    if len(n_gram) == 2:
        return n_gram in bigrams.keys()

    if len(n_gram) == 3:
        return n_gram in trigrams.keys()


def get_discounted_MLE_prob(n_gram, unigrams, bigrams, trigrams):
    if len(n_gram) == 3:
        numer = trigrams[n_gram] - DISCOUNT
        denom = bigrams[tuple(n_gram[:-1])]

    if len(n_gram) == 2:
        numer = bigrams[n_gram] - DISCOUNT
        denom = unigrams[tuple(n_gram[:-1])]

    if len(n_gram) == 1:
        numer = unigrams[n_gram]
        denom = sum(unigrams.values()) - unigrams[(START,)]

    return numer / denom


def get_backoff_denom(history_n_gram, unigrams, bigrams, trigrams, n_grams_to_probs, history_to_alphas, history_to_denoms):
    if history_n_gram in history_to_denoms:
        return history_to_denoms[history_n_gram]

    existing = set()
    for n_gram, count in get_n_gram_counts_for_some_history(history_n_gram, bigrams, trigrams):
        existing.add(tuple(n_gram[-1],))
    w_s = set(unigrams.keys()).difference(existing)

    if len(history_n_gram) == 2:
        denom = 0
        for w in w_s:
            denom += get_prob((history_n_gram[1], w[0]), unigrams, bigrams, trigrams, n_grams_to_probs, history_to_alphas, history_to_denoms)

        history_to_denoms[history_n_gram] = denom
        return denom

    if len(history_n_gram) == 1:
        denom = 0
        for w in w_s:
            if n_gram_seen_in_training(w, unigrams, bigrams, trigrams):
                denom += get_discounted_MLE_prob(w, unigrams, bigrams, trigrams)

        history_to_denoms[history_n_gram] = denom
        return denom


def get_alpha(history_n_gram, unigrams, bigrams, trigrams, history_to_alphas):
    if history_n_gram in history_to_alphas:
        return history_to_alphas[history_n_gram]

    ret = 0
    for n_gram, count in get_n_gram_counts_for_some_history(history_n_gram, bigrams, trigrams):
        if len(history_n_gram) == 2:
            ret += (count - DISCOUNT) / bigrams[history_n_gram]
        if len(history_n_gram) == 1:
            ret += (count - DISCOUNT) / unigrams[history_n_gram]
    ret = 1 - ret

    history_to_alphas[history_n_gram] = ret
    return ret


def get_n_gram_counts_for_some_history(history, bigrams, trigrams):
    if len(history) == 2:
        for n_gram, count in trigrams.items():
            if all(k1 == k2 or k2 is None for k1, k2 in zip(n_gram, (history[0], history[1], None))):
                yield n_gram, count

    if len(history) == 1:
        for n_gram, count in bigrams.items():
            if all(k1 == k2 or k2 is None for k1, k2 in zip(n_gram, (history[0], None))):
                yield n_gram, count


if __name__ == '__main__':
    main()