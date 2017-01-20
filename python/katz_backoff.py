import math
import n_gram_utils
from collections import Counter


# start symbol
START = '<s>'
# stop symbol
STOP = '</s>'
# unk symbol
UNK = '<UNK>'
# Katz back-off count discount hyper-parameter
KATZ_DISCOUNT = 0.5


def main():
    unigrams, bigrams, trigrams, unigrams_to_w_to_counts, bigrams_to_w_to_counts = train('data/brown_train.txt')
    model = (unigrams, bigrams, trigrams)

    def get_log_prob(n_gram, model, unigrams_to_w_to_counts, bigrams_to_w_to_counts, n_grams_to_probs, history_to_alphas, history_to_denoms):
        prob = get_prob(n_gram, model, unigrams_to_w_to_counts, bigrams_to_w_to_counts, n_grams_to_probs, history_to_alphas, history_to_denoms)
        log_prob = math.log(prob, 2)
        return log_prob

    # returns the probability of a specified n-gram in the model
    def get_prob(n_gram, model, unigrams_to_w_to_counts, bigrams_to_w_to_counts, n_grams_to_probs, history_to_alphas, history_to_denoms):

        # in the case that we've calculated this probability before
        if n_gram in n_grams_to_probs:
            return n_grams_to_probs[n_gram]

        # if we're trying to find the probability of the empty sequence, it's just 0
        # honestly, this shouldn't happen too often
        if len(n_gram) == 0:
            print('CALLED GET PROB WITH SEQUENCE LENGTH OF 0')
            return 0

        # if our n-gram appeared in our training data, that means we can just go grab the MLE
        # also, if our n-gram is a uni-gram, we can just go grab it's probability
        # if n_gram_seen_in_training(n_gram, model) or len(n_gram) == 1:
        if len(n_gram) == 1 or n_gram in model[len(n_gram) - 1]:
            prob = get_discounted_MLE_prob(n_gram, model)
            return prob

        # here, we need to perform the back-off
        next_gram = n_gram[1:]
        history_n_gram = n_gram[:-1]
        numer = get_prob(next_gram, model, unigrams_to_w_to_counts, bigrams_to_w_to_counts, n_grams_to_probs, history_to_alphas, history_to_denoms)
        denom = get_backoff_denom(history_n_gram, model, unigrams_to_w_to_counts, bigrams_to_w_to_counts, n_grams_to_probs, history_to_alphas, history_to_denoms)
        alpha = get_alpha(history_n_gram, model, unigrams_to_w_to_counts, bigrams_to_w_to_counts, history_to_alphas)

        prob = alpha * numer / denom
        n_grams_to_probs[n_gram] = prob

        return prob

    def get_discounted_MLE_prob(n_gram, model):
        if len(n_gram) == 1:
            numer = model[0][n_gram]
            denom = sum(model[0].values()) - model[0][(n_gram_utils.START,)]
        else:
            numer = model[len(n_gram) - 1][n_gram] - KATZ_DISCOUNT
            denom = model[len(n_gram) - 2][n_gram[:-1]]

        return numer / denom

    def get_backoff_denom(history_n_gram, model, unigrams_to_w_to_counts, bigrams_to_w_to_counts, n_grams_to_probs, history_to_alphas, history_to_denoms):
        if history_n_gram in history_to_denoms:
            return history_to_denoms[history_n_gram]

        if len(history_n_gram) == 2:
            curr_sum = 0
            if history_n_gram in bigrams_to_w_to_counts:
                for word, count in bigrams_to_w_to_counts[history_n_gram].items():
                    curr_sum += get_discounted_MLE_prob((history_n_gram[-1], word), model)

            history_to_denoms[history_n_gram] = 1 - curr_sum
            return 1 - curr_sum

        if len(history_n_gram) == 1:
            curr_sum = 0
            for word, count in unigrams_to_w_to_counts[history_n_gram].items():
                curr_sum += get_discounted_MLE_prob((history_n_gram[-1], word), model)

            history_to_denoms[history_n_gram] = 1 - curr_sum
            return 1 - curr_sum

    def get_alpha(history_n_gram, model, unigrams_to_w_to_counts, bigrams_to_w_to_counts, history_to_alphas):
        if history_n_gram in history_to_alphas:
            return history_to_alphas[history_n_gram]

        ret = 0
        if len(history_n_gram) == 2:
            if history_n_gram in bigrams_to_w_to_counts:
                for word, count in bigrams_to_w_to_counts[history_n_gram].items():
                    ret += (count - KATZ_DISCOUNT) / model[1][history_n_gram]

            ret = 1 - ret

        if len(history_n_gram) == 1:
            ret = 0
            for word, count in unigrams_to_w_to_counts[history_n_gram].items():
                ret += (count - KATZ_DISCOUNT) / model[0][history_n_gram]

            ret = 1 - ret

        history_to_alphas[history_n_gram] = ret
        return ret

    perplexity = eval_model('data/brown_dev.txt', model, get_log_prob, unigrams_to_w_to_counts, bigrams_to_w_to_counts, dict(), dict(), dict())
    print('perplexity is: ' + str(perplexity))


def eval_model(filename, model, log_prob_func, unigrams_to_w_to_counts, bigrams_to_w_to_counts, n_grams_to_probs, history_to_alphas, history_to_denoms):
    print('evaluating model')

    log_prob_sum = 0
    file_word_count = 0

    with open(filename) as f:
        for line in f:
            log_prob, num_tokens = eval_sentence(line, model, log_prob_func, unigrams_to_w_to_counts, bigrams_to_w_to_counts, n_grams_to_probs, history_to_alphas, history_to_denoms)
            #print('got a sentence prob: ' + str(2**(log_prob)))
            log_prob_sum += log_prob
            file_word_count += num_tokens
        f.close()

    print('finished evaluating!')
    average_log_prob = log_prob_sum / file_word_count
    perplexity = 2**(-average_log_prob)
    return perplexity


# returns log probability of a sentence and how many tokens were in the sentence
def eval_sentence(sentence, model, log_prob_func, unigrams_to_w_to_counts, bigrams_to_w_to_counts, n_grams_to_probs, history_to_alphas, history_to_denoms):
    tokens = [token if (token,) in model[0] else n_gram_utils.UNK for token in sentence.split()]
    num_tokens = len(tokens) + 1
    tokens.insert(0, n_gram_utils.START)
    tokens.insert(0, n_gram_utils.START)
    tokens.append(n_gram_utils.STOP)

    log_prob_sum = 0
    for i in range(len(tokens) - 2):
        n_gram = tuple(tokens[i:i+3])
        #print('evaluating the n-gram: ' + str(n_gram))
        next_prob = log_prob_func(n_gram, model, unigrams_to_w_to_counts, bigrams_to_w_to_counts, n_grams_to_probs, history_to_alphas, history_to_denoms)
        #print('got the prob: ' + str(next_prob))
        log_prob_sum += next_prob

    return log_prob_sum, num_tokens


def train(filename):
    print('training...')

    # initializing empty Counter objects to store the n-grams
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    bigrams_to_w_to_counts = dict()
    unigrams_to_w_to_counts = dict()

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
        add_n_gram_counts(1, unigrams, tokens, None)

    # the set of all words that have a count of 1
    unks = set()
    for unigram, count in unigrams.items():
        if count == 1:
            unks.add(unigram[0])

    for word in unks:
        del unigrams[(word,)]

    unigrams[(UNK,)] = len(unks)

    # creating the bigram and trigram counts
    for line in lines:
        tokens = [token if token not in unks else UNK for token in line.split()]
        tokens.insert(0, START)
        tokens.insert(0, START)
        tokens.append(STOP)
        add_n_gram_counts(2, bigrams, tokens, unigrams_to_w_to_counts)
        add_n_gram_counts(3, trigrams, tokens, bigrams_to_w_to_counts)

    print('finished training!')
    return unigrams, bigrams, trigrams, unigrams_to_w_to_counts, bigrams_to_w_to_counts


# adds the n-grams to the specified Counter from the specified tokens
def add_n_gram_counts(n, n_grams, tokens, cache):
    for i in range(len(tokens) - (n - 1)):
        n_gram = tuple(tokens[i:i+n])
        n_grams[n_gram] += 1

        if n > 1:
            if n_gram[:-1] not in cache:
                cache[n_gram[:-1]] = Counter()
            cache[n_gram[:-1]][n_gram[-1]] +=1

    return n_grams


if __name__ == '__main__':
    main()