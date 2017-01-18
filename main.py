import math
from n_gram import NGram
from collections import Counter


START = '<s>'
STOP = '</s>'
DISCOUNT = 0.5


def main():
    unigrams, bigrams, trigrams = train('data/brown_train.txt')
    #eval_model('data/brown_dev.txt', unigrams, bigrams, trigrams)
    #test_unigrams, test_bigrams, test_trigrams = train('data/brown_dev.txt')
    sentence = 'The marines let him advance .'
    sentence_perplexity = eval_sentence(sentence, unigrams, bigrams, trigrams) / len(sentence.split())
    print(str(sentence_perplexity))
    #print(str(get_prob(NGram(('Thus', ',', 'while')), unigrams, bigrams, trigrams, dict(), dict())))
    #print(str(get_backoff_denom(NGram((',',)), unigrams, bigrams, trigrams, dict(), dict())))


# returns the perplexity
def eval_model(filename, unigrams, bigrams, trigrams):
    print('evaluating model')

    with open(filename) as f:
        overall_prob = 1

        for line in f:
            prob = eval_sentence(line, unigrams, bigrams, trigrams)
            #print('got a prob ' + str(prob))
            overall_prob *= prob

        f.close()
        print('finished evaluating!')
        return overall_prob


def train(filename):
    print('training...')

    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()

    with open(filename) as f:
        for line in f:
            tokens = line.split()
            tokens.insert(0, START)
            tokens.insert(0, START)
            tokens.append(STOP)
            get_n_gram_counts(1, unigrams, tokens)
            get_n_gram_counts(2, bigrams, tokens)
            get_n_gram_counts(3, trigrams, tokens)

    print('finished training!')

    return unigrams, bigrams, trigrams


def get_n_gram_counts(n, n_grams, tokens):
    # creating n-grams
    for i in range(0, len(tokens) - (n - 1)):
        n_grams[NGram(tuple(tokens[i:i+n]))] += 1

    return n_grams


# returns LOG probability of a sentence
def eval_sentence(sentence, unigrams, bigrams, trigrams):
    tokens = sentence.split()
    tokens.insert(0, START)
    tokens.insert(0, START)
    tokens.append(STOP)

    prob = 1
    history_to_alphas = dict()
    history_to_denoms = dict()
    n_grams_to_probs = dict()

    for i in range(len(tokens) - 2):
        n_gram = NGram((tuple(tokens[i:i+3])))
        print('getting the probability for the n_gram ' + str(n_gram.seq))
        next_prob = get_prob(n_gram, unigrams, bigrams, trigrams, n_grams_to_probs, history_to_alphas, history_to_denoms)
        print('got the probability: ' + str(next_prob))
        prob *= next_prob
        print('evaluated an n-gram')

    log_prob = math.log(prob, 2)
    return log_prob


def get_prob(n_gram, unigrams, bigrams, trigrams, n_grams_to_probs, history_to_alphas, history_to_denoms):
    #print('called get_prob with n-gram ' + str(n_gram.seq))
    if n_gram in n_grams_to_probs:
        return n_grams_to_probs[n_gram]
    if n_gram.n == 0:
        return 0

    if n_gram_exists(n_gram, unigrams, bigrams, trigrams) or n_gram.n == 1:
        return get_discounted_MLE_prob(n_gram, unigrams, bigrams, trigrams)

    next_gram = NGram(n_gram.seq[1:])
    history_n_gram = NGram(n_gram.seq[:-1])
    numer = get_prob(next_gram, unigrams, bigrams, trigrams, n_grams_to_probs, history_to_alphas, history_to_denoms)
    denom = get_backoff_denom(history_n_gram, unigrams, bigrams, trigrams, n_grams_to_probs, history_to_alphas, history_to_denoms)
    alpha = get_alpha(history_n_gram, unigrams, bigrams, trigrams, history_to_alphas)
    '''if denom == 0:
        #print('got a 0 denom for the n_gram ' + str(n_gram.seq))'''

    prob = alpha * numer / denom
    n_grams_to_probs[n_gram] = prob

    return prob


def n_gram_exists(n_gram, unigrams, bigrams, trigrams):
    if n_gram.n == 1:
        return n_gram in unigrams.keys()

    if n_gram.n == 2:
        return n_gram in bigrams.keys()

    if n_gram.n == 3:
        return n_gram in trigrams.keys()


def get_discounted_MLE_prob(n_gram, unigrams, bigrams, trigrams):
    if n_gram.n == 3:
        numer = trigrams[n_gram] - DISCOUNT
        denom = bigrams[NGram(n_gram.seq[:-1])]

    if n_gram.n == 2:
        numer = bigrams[n_gram] - DISCOUNT
        denom = unigrams[NGram(n_gram.seq[:-1])]

    if n_gram.n == 1:
        numer = unigrams[n_gram]
        denom = sum(unigrams.values())

    # neither of these should be 0, ever
    return numer / denom


# TODO: implement this
def get_backoff_denom(history_n_gram, unigrams, bigrams, trigrams, n_grams_to_probs, history_to_alphas, history_to_denoms):
    #print('called get_backoff_denom with n-gram ' + str(history_n_gram.seq))
    #print('history_n_gram size is ' + str(history_n_gram.n))
    if history_n_gram in history_to_denoms:
        return history_to_denoms[history_n_gram]

    existing = set()
    for n_gram, count in get_n_gram_counts_for_some_history(history_n_gram.seq, bigrams, trigrams):
        existing.add(NGram((n_gram.seq[-1],)))

    w_s = set(unigrams.keys()).difference(existing)
    #print('w\'s is of size ' + str(len(w_s)))

    if history_n_gram.n == 2:
        #print('n gram is of size 2')
        sum = 0
        for w in w_s:
            sum += get_prob(NGram((history_n_gram.seq[1], w.seq[0])), unigrams, bigrams, trigrams, n_grams_to_probs, history_to_alphas, history_to_denoms)

        history_to_denoms[history_n_gram] = sum
        return sum

    if history_n_gram.n == 1:
        sum = 0
        for w in w_s:
            if n_gram_exists(w, unigrams, bigrams, trigrams):
                sum += get_discounted_MLE_prob(w, unigrams, bigrams, trigrams)

        history_to_denoms[history_n_gram] = sum
        return sum


def get_alpha(history_n_gram, unigrams, bigrams, trigrams, history_to_alphas):
    if history_n_gram in history_to_alphas:
        return history_to_alphas[history_n_gram]

    ret = 0
    for n_gram, count in get_n_gram_counts_for_some_history(history_n_gram.seq, bigrams, trigrams):
        if history_n_gram.n == 2:
            ret += (count - DISCOUNT) / bigrams[history_n_gram]
        if history_n_gram.n == 1:
            ret += (count - DISCOUNT) / unigrams[history_n_gram]
    ret = 1 - ret

    history_to_alphas[history_n_gram] = ret
    return ret


def get_n_gram_counts_for_some_history(history, bigrams, trigrams):
    if len(history) == 2:
        for n_gram, count in trigrams.items():
            if all(k1 == k2 or k2 is None for k1, k2 in zip(n_gram.seq, (history[0], history[1], None))):
                yield n_gram, count

    if len(history) == 1:
        for n_gram, count in bigrams.items():
            if all(k1 == k2 or k2 is None for k1, k2 in zip(n_gram.seq, (history[0], None))):
                yield n_gram, count


if __name__ == '__main__':
    main()