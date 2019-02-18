#!/usr/bin/env python3

import math
import argparse

import config as cfg
import n_gram as ng


def main(args):
    def get_log_prob(n_gram, model, caches):
        '''This is a nested method that will be passed to eval_model().

        n_gram is a tuple of strings that represents an n-gram
        model is a tuple of Counter objects (unigrams, bigrams, trigrams) that map from n-grams to
        counts caches is a tuple of dicts (unigrams_to_probs, bigrams_to_probs,
            n_grams_to_interpolated_probs) that memoize the return values of various function calls
        '''

        if n_gram in caches[2]:
            return caches[2][n_gram]

        vocab_size = len(model[0]) - 1

        # uni-gram part
        if n_gram[2:] in caches[0]:
            unigram_part = caches[0][n_gram[2:]]
        else:
            uni_numer = model[0][n_gram[2:]]
            uni_denom = sum(model[0].values()) - model[0][(cfg.START,)]
            unigram_part = cfg.LAMBDA_1 * uni_numer / uni_denom
            caches[0][n_gram[2:]] = unigram_part

        # bi-gram part
        if n_gram[1:] in caches[1]:
            bigram_part = caches[1][n_gram[1:]]
        else:
            bi_numer = model[1][n_gram[1:]]
            bi_denom = model[0][n_gram[1:2]]
            bigram_part = cfg.LAMBDA_2 * bi_numer / bi_denom
            caches[1][n_gram[1:]] = bigram_part

        # tri-gram part
        tri_numer = model[2][n_gram]
        tri_denom = model[1][n_gram[:2]]
        if tri_denom == 0:
            trigram_part = cfg.LAMBDA_3 * 1 / vocab_size
        else:
            trigram_part = cfg.LAMBDA_3 * tri_numer / tri_denom

        prob = trigram_part + bigram_part + unigram_part
        log_prob = math.log(prob, 2)
        caches[2][n_gram] = log_prob
        return log_prob

    print('training interpolated n-gram model on', cfg.TRAIN, '...')
    print('UNK_THRESHOLD is:', cfg.UNK_THRESHOLD)

    unigrams, bigrams, trigrams = ng.train(cfg.TRAIN)
    model = (unigrams, bigrams, trigrams)

    print('vocab size is', len(unigrams) - 1)
    print('num tokens is', sum(model[0].values()) - model[0][(cfg.START,)])
    print('num UNK is', model[0][(cfg.UNK,)])
    print()

    if args.tune:
        print('tuning lambda hyperparemeters for interpolated n-gram model ...')
        print()
        for lambda_1 in cfg.LAMBDA_1s:
            for lambda_2 in cfg.LAMBDA_2s:
                cfg.LAMBDA_1 = lambda_1
                cfg.LAMBDA_2 = lambda_2
                if lambda_1 + lambda_2 <= 0.9:
                    lambda_3 = 1 - lambda_1 - lambda_2
                    cfg.LAMBDA_3 = lambda_3
                    print('lambda 1:', cfg.LAMBDA_1, 'lambda 2:', cfg.LAMBDA_2, 'lambda 3',
                          cfg.LAMBDA_3)
                    # (unigrams_to_probs, bigrams_to_probs, n_grams_to_interpolated_probs)
                    caches = (dict(), dict(), dict())
                    print('evaluating on', cfg.TRAIN, 'train set ...')
                    perplexity = ng.eval_model(3, cfg.TRAIN, model, get_log_prob, caches)
                    print('perplexity:', perplexity)
                    print('evaluating on', cfg.DEV, 'dev set ...')
                    perplexity = ng.eval_model(3, cfg.DEV, model, get_log_prob, caches)
                    print('perplexity:', perplexity)
                    if args.test:
                        print('evaluating on', cfg.TEST, 'test set ...')
                        perplexity = ng.eval_model(3, cfg.TEST, model, get_log_prob, caches)
                        print('perplexity:', perplexity)
                    print()
    else:
        print('lambda 1:', cfg.LAMBDA_1, 'lambda 2:', cfg.LAMBDA_2, 'lambda 3', cfg.LAMBDA_3)
        # (unigrams_to_probs, bigrams_to_probs, n_grams_to_interpolated_probs)
        caches = (dict(), dict(), dict())

        print('evaluating on', cfg.TRAIN, 'train set ...')
        perplexity = ng.eval_model(3, cfg.TRAIN, model, get_log_prob, caches)
        print('perplexity:', perplexity)
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
    args = parser.parse_args()
    main(args)
