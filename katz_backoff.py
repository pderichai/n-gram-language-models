import math
import n_gram_utils


# Katz back-off discount hyper-parameter
KATZ_DISCOUNT = 0.5


def main():
    model = n_gram_utils.train('data/brown_train.txt')


    def get_log_prob(n_gram, model, n_grams_to_probs, history_to_alphas, history_to_denoms):
        prob = get_prob(n_gram, model, n_grams_to_probs, history_to_alphas, history_to_denoms)
        log_prob = math.log(prob, 2)
        return log_prob


    # returns the probability of a specified n-gram in the model
    def get_prob(n_gram, model, n_grams_to_probs, history_to_alphas, history_to_denoms):
        #print('getting the prob of ' + str(n_gram))

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
        if n_gram_seen_in_training(n_gram, model) or len(n_gram) == 1:
            prob = get_discounted_MLE_prob(n_gram, model)
            return prob

        # here, we need to perform the back-off
        next_gram = tuple(n_gram[1:])
        history_n_gram = tuple(n_gram[:-1])
        numer = get_prob(next_gram, model, n_grams_to_probs, history_to_alphas, history_to_denoms)
        denom = get_backoff_denom(history_n_gram, model, n_grams_to_probs, history_to_alphas, history_to_denoms)
        alpha = get_alpha(history_n_gram, model, history_to_alphas)

        prob = alpha * numer / denom
        n_grams_to_probs[n_gram] = prob

        return prob


    # given a specified n-gram, returns whether or not it was seen in the training data
    def n_gram_seen_in_training(n_gram, model):
        if len(n_gram) == 1:
            return n_gram in model[0]

        if len(n_gram) == 2:
            return n_gram in model[1]

        if len(n_gram) == 3:
            return n_gram in model[2]


    def get_discounted_MLE_prob(n_gram, model):
        if len(n_gram) == 3:
            numer = model[2][n_gram] - KATZ_DISCOUNT
            denom = model[1][tuple(n_gram[:-1])]

        if len(n_gram) == 2:
            numer = model[1][n_gram] - KATZ_DISCOUNT
            denom = model[0][tuple(n_gram[:-1])]

        if len(n_gram) == 1:
            numer = model[0][n_gram]
            denom = sum(model[0].values()) - model[0][(n_gram_utils.START,)]

        return numer / denom


    def get_backoff_denom(history_n_gram, model, n_grams_to_probs, history_to_alphas, history_to_denoms):
        if history_n_gram in history_to_denoms:
            return history_to_denoms[history_n_gram]

        existing = set()
        for n_gram, count in get_n_gram_counts_for_some_history(history_n_gram, model[1], model[2]):
            existing.add(tuple(n_gram[-1],))
        w_s = set(model[0].keys()).difference(existing)

        if len(history_n_gram) == 2:
            denom = 0
            for w in w_s:
                denom += get_prob((history_n_gram[1], w[0]), model, n_grams_to_probs, history_to_alphas, history_to_denoms)

            history_to_denoms[history_n_gram] = denom
            return denom

        if len(history_n_gram) == 1:
            denom = 0
            for w in w_s:
                if n_gram_seen_in_training(w, model):
                    denom += get_discounted_MLE_prob(w, model)

            history_to_denoms[history_n_gram] = denom
            return denom


    def get_alpha(history_n_gram, model, history_to_alphas):
        if history_n_gram in history_to_alphas:
            return history_to_alphas[history_n_gram]

        ret = 0
        for n_gram, count in get_n_gram_counts_for_some_history(history_n_gram, model[1], model[2]):
            if len(history_n_gram) == 2:
                ret += (count - KATZ_DISCOUNT) / model[1][history_n_gram]
            if len(history_n_gram) == 1:
                ret += (count - KATZ_DISCOUNT) / model[0][history_n_gram]
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


    perplexity = eval_model('data/brown_dev.txt', model, get_log_prob, dict(), dict(), dict())


def eval_model(filename, model, log_prob_func, n_grams_to_probs, history_to_alphas, history_to_denoms):
    print('evaluating model')

    log_prob_sum = 0
    file_word_count = 0

    with open(filename) as f:
        for line in f:
            prob, num_tokens = eval_sentence(line, model, log_prob_func, n_grams_to_probs, history_to_alphas, history_to_denoms)
            log_prob_sum += prob
            file_word_count += num_tokens
        f.close()

    print('finished evaluating!')
    average_log_prob = log_prob_sum / file_word_count
    perplexity = 2**(-average_log_prob)
    return perplexity


# returns log probability of a sentence and how many tokens were in the sentence
def eval_sentence(sentence, model, log_prob_func, n_grams_to_probs, history_to_alphas, history_to_denoms):
    tokens = [token if (token,) in model[0] else n_gram_utils.UNK for token in sentence.split()]
    num_tokens = len(tokens) + 1
    tokens.insert(0, n_gram_utils.START)
    tokens.insert(0, n_gram_utils.START)
    tokens.append(n_gram_utils.STOP)

    log_prob_sum = 0
    for i in range(len(tokens) - 2):
        n_gram = tuple(tokens[i:i+3])
        print('evaluating the n-gram: ' + str(n_gram))
        next_prob = log_prob_func(n_gram, model, n_grams_to_probs, history_to_alphas, history_to_denoms)
        print('got the prob: ' + str(next_prob))
        log_prob_sum += next_prob

    return log_prob_sum, num_tokens


if __name__ == '__main__':
    main()