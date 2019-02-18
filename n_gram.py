#!/usr/bin/env python3

import math
import argparse
from collections import Counter

import config as cfg


def main(args):
    def get_log_prob(n_gram, model, caches):
        """
        This is a nested method that will be passed to eval_model().

        Args:
            n_gram (tuple of str): represents an n-gram
            model (tuple of Counter): (unigrams, bigrams, trigrams) maps from
                n-grams to counts
            caches (tuple of dict): (unigrams_to_probs, bigrams_to_probs,
                trigrams_to_probs) dicts that cache the probabilities when they
                are calculated

        Returns:
            The log probability of the specified n-gram under the model.
        """
        # unigram
        if len(n_gram) == 1:
            if n_gram in caches[0]:
                log_prob = caches[0][n_gram]
            else:
                uni_numer = model[0][n_gram]
                uni_denom = sum(model[0].values()) - model[0][(cfg.START,)]
                log_prob = math.log(uni_numer / uni_denom, 2)
                caches[0][n_gram] = log_prob

        # bigram
        if len(n_gram) == 2:
            if n_gram in caches[1]:
                log_prob = caches[1][n_gram]
            else:
                bi_numer = model[1][n_gram]
                bi_denom = model[0][n_gram[:1]]
                if bi_numer == 0:
                    return float('-inf')
                log_prob = math.log(bi_numer / bi_denom, 2)
                caches[1][n_gram] = log_prob

        # trigram
        if len(n_gram) == 3:
            if n_gram in caches[2]:
                log_prob = caches[2][n_gram]
            else:
                tri_numer = model[2][n_gram]
                tri_denom = model[1][n_gram[:2]]
                if tri_denom == 0:
                    vocab_size = len(model[0]) - 1
                    return math.log(1 / vocab_size, 2)
                if tri_numer == 0:
                    return float('-inf')
                log_prob = math.log(tri_numer / tri_denom, 2)
                caches[2][n_gram] = log_prob

        return log_prob

    print('training n-gram model on', cfg.TRAIN, '...')
    print('UNK_THRESHOLD is:', cfg.UNK_THRESHOLD)

    unigrams, bigrams, trigrams = train(cfg.TRAIN)
    model = (unigrams, bigrams, trigrams)

    print('vocab size is', len(unigrams) - 1)
    print('num tokens is', sum(model[0].values()) - model[0][(cfg.START,)])
    print('num UNK is', model[0][(cfg.UNK,)])
    print()

    # (unigrams_to_probs, bigrams_to_probs, trigrams_to__probs)
    caches = (dict(), dict(), dict())

    for n in range(1, 4):
        print('evaluating', str(n) + '-gram model ...')
        print('evaluating on', cfg.TRAIN,'train set ...')
        perplexity = eval_model(n, cfg.TRAIN, model, get_log_prob, caches)
        print('perplexity:', str(perplexity))
        print('evaluating on', cfg.DEV,'dev set ...')
        perplexity = eval_model(n, cfg.DEV, model, get_log_prob, caches)
        print('perplexity:', str(perplexity))
        if args.test:
            print('evaluating on', cfg.TEST,'test set ...')
            perplexity = eval_model(n, cfg.TEST, model, get_log_prob, caches)
            print('perplexity:', str(perplexity))
        print()


def train(filename):
    """
    Trains an n-gram model using the specified corpus.

    Args:
        filename (str): the filename of the corpus

    Returns:
        A tuple of (Counter, Counter, Counter) which are Counter objects from
            (tuple, int) that are the counts of the unigrams, bigrams, and
            trigrams respectively.
    """

    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()

    lines = list()
    with open(filename) as f:
        for line in f:
            lines.append(line)

    # creating the uni-gram counts
    for line in lines:
        tokens = line.split()
        tokens.insert(0, cfg.START)
        tokens.insert(0, cfg.START)
        tokens.append(cfg.STOP)
        add_n_gram_counts(1, unigrams, tokens)

    # the set of all uni-grams that have a count less than UNK_THRESHOLD
    unks = set()
    num_unks = 0
    for unigram, count in unigrams.items():
        if count < cfg.UNK_THRESHOLD:
            unks.add(unigram[0])
            num_unks += count

    for word in unks:
        del unigrams[(word,)]

    unigrams[(cfg.UNK,)] = num_unks

    # creating the bigram and trigram counts
    for line in lines:
        tokens = [token if token not in unks else cfg.UNK for token in line.split()]
        tokens.insert(0, cfg.START)
        tokens.insert(0, cfg.START)
        tokens.append(cfg.STOP)
        add_n_gram_counts(2, bigrams, tokens)
        add_n_gram_counts(3, trigrams, tokens)

    return unigrams, bigrams, trigrams


def add_n_gram_counts(n, n_grams, tokens):
    """Adds the n-grams to the specified Counter from the specified tokens."""
    for i in range(len(tokens) - (n - 1)):
        n_grams[tuple(tokens[i:i+n])] += 1

    return n_grams


def eval_model(n, filename, model, log_prob_func, caches):
    """Returns the perplexity of the model on a specified test set."""

    log_prob_sum = 0
    file_word_count = 0

    with open(filename) as f:
        for line in f:
            prob, num_tokens = eval_sentence(n, line, model, log_prob_func, caches)
            log_prob_sum += prob
            file_word_count += num_tokens
        f.close()

    average_log_prob = log_prob_sum / file_word_count
    perplexity = 2**(-average_log_prob)
    return perplexity


def eval_sentence(n, sentence, model, log_prob_func, caches):
    """
    Returns log probability of a sentence and how many tokens were in the
        sentence.
    """

    tokens = [token if (token,) in model[0] else cfg.UNK for token in sentence.split()]
    num_tokens = len(tokens) + 1
    for _ in range(1, n):
        tokens.insert(0, cfg.START)
    tokens.append(cfg.STOP)

    log_prob_sum = 0
    for i in range(len(tokens) - (n - 1)):
        n_gram = tuple(tokens[i:i+n])
        next_prob = log_prob_func(n_gram, model, caches)
        log_prob_sum += next_prob

    return log_prob_sum, num_tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='Evaluate on the test data.',
            action='store_true')
    args = parser.parse_args()
    main(args)
