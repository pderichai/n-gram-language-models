#!/usr/bin/env python3

import math
import argparse

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
import n_gram as ng


def main(args):
    def get_log_prob(n_gram, model, caches):
        '''This is a nested method that will be passed to eval_model().

        n_gram (tuple of str): represents an n-gram
        model (tuple of Counter): (unigrams, bigrams, trigrams) maps from n-grams to counts
        caches (tuple of dict): (unigrams_to_probs, bigrams_to_probs, trigrams_to_probs) dicts
            that cache the probabilities when they are calculated
        '''
        # tri-gram
        if len(n_gram) == 3:
            if n_gram in caches[2]:
                log_prob = caches[2][n_gram]
            else:
                vocab_size = len(model[0]) - 1
                tri_numer = model[2][n_gram] + cfg.K
                tri_denom = model[1][n_gram[:2]] + cfg.K * vocab_size
                if tri_denom == 0:
                    return math.log(1 / vocab_size)
                if tri_numer == 0:
                    return float('-inf')
                log_prob = math.log(tri_numer / tri_denom, 2)
                caches[2][n_gram] = log_prob

        return log_prob

    print('training add-K n-gram model on', cfg.TRAIN, '...')
    print('UNK_THRESHOLD is', cfg.UNK_THRESHOLD)

    unigrams, bigrams, trigrams = ng.train(cfg.TRAIN)
    model = (unigrams, bigrams, trigrams)

    print('vocab size is', len(unigrams) - 1)
    print('num tokens is', sum(model[0].values()) - model[0][(cfg.START,)])
    print('num UNK is', model[0][(cfg.UNK,)])
    print()

    if args.tune:
        # coarse-tuning
        print('coarse-tuning K ...')
        dev_perplexities = []
        train_perplexities = []
        K_vals = [10**exp for exp in range(-5, 3)]
        for K in K_vals:
            cfg.K = K
            print('K is', cfg.K)
            caches = (dict(), dict(), dict())

            print('evaluating on', cfg.TRAIN, 'train set ...')
            perplexity = ng.eval_model(3, cfg.TRAIN, model, get_log_prob, caches)
            train_perplexities.append(perplexity)
            print('perplexity:', str(perplexity))
            print('evaluating on', cfg.DEV, 'dev set ...')
            perplexity = ng.eval_model(3, cfg.DEV, model, get_log_prob, caches)
            dev_perplexities.append(perplexity)
            print('perplexity:', perplexity)
            print()
        if args.plots:
            plt.figure()
            plt.plot(K_vals, train_perplexities, '-o', label='train')
            plt.plot(K_vals, dev_perplexities, '-o', label='dev')
            plt.legend()
            plt.title('Add-K Coarse-Tuning')
            plt.xlabel('K value')
            plt.ylabel('perplexity')
            plt.xscale('log')
            plt.savefig('add-k-coarse-tuning.png')

        # fine-tuning
        print('fine-tuning K ...')
        dev_perplexities = []
        train_perplexities = []
        K_vals = np.linspace(1e-4, 1e-2, 11)
        for K in K_vals:
            cfg.K = K
            print('K is', cfg.K)
            caches = (dict(), dict(), dict())

            print('evaluating on', cfg.TRAIN, 'train set ...')
            perplexity = ng.eval_model(3, cfg.TRAIN, model, get_log_prob, caches)
            train_perplexities.append(perplexity)
            print('perplexity:', str(perplexity))
            print('evaluating on', cfg.DEV, 'dev set ...')
            perplexity = ng.eval_model(3, cfg.DEV, model, get_log_prob, caches)
            dev_perplexities.append(perplexity)
            print('perplexity:', perplexity)
            print()
        if args.plots:
            plt.figure()
            plt.plot(K_vals, train_perplexities, '-o', label='train')
            plt.plot(K_vals, dev_perplexities, '-o', label='dev')
            plt.legend()
            plt.title('Add-K Fine-Tuning')
            plt.xlabel('K value')
            plt.ylabel('perplexity')
            plt.savefig('add-k-fine-tuning.png')
    else:
        print('K is', cfg.K)
        caches = (dict(), dict(), dict())

        print('evaluating on', cfg.TRAIN, 'train set ...')
        perplexity = ng.eval_model(3, cfg.TRAIN, model, get_log_prob, caches)
        print('perplexity:', str(perplexity))
        print('evaluating on', cfg.DEV, 'dev set ...')
        perplexity = ng.eval_model(3, cfg.DEV, model, get_log_prob, caches)
        print('perplexity:', perplexity)
        if args.test:
            print('evaluating on', cfg.TEST, 'test set ...')
            perplexity = ng.eval_model(3, cfg.TEST, model, get_log_prob, caches)
            print('perplexity:', perplexity)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='Evaluate on the test data.',
            action='store_true')
    parser.add_argument('--tune', help='Grid search over hyper-parameters.',
            action='store_true')
    parser.add_argument('--plots', help='Generate plots of hyper-parameter tuning.',
            action='store_true')
    args = parser.parse_args()
    main(args)
