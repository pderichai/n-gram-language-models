import math
import n_gram_utils


# linear interpolation hyper-parameters
LAMBDA_1 = 0.2
LAMBDA_2 = 0.3
LAMBDA_3 = 0.5


# just the main method
def main():
    model = n_gram_utils.train('data/brown_train.txt')


    # returns the log probability of a specified n-gram
    def get_log_prob(n_gram, model):
        # tri-gram part
        tri_numer = model[2][n_gram]
        tri_denom = model[1][tuple(n_gram[:-1])]
        trigram_part = 0
        if tri_denom != 0:
            trigram_part = LAMBDA_1 * tri_numer / tri_denom

        # bi-gram part
        bi_numer = model[1][tuple(n_gram[:-1])]
        bi_denom = model[0][tuple(n_gram[:-2])]
        bigram_part = 0
        if bi_denom != 0:
            bigram_part = LAMBDA_2 * bi_numer / bi_denom

        # uni-gram part
        uni_numer = model[0][tuple(n_gram[:-2])]
        uni_denom = sum(model[0].values()) - model[0][(n_gram_utils.START,)]
        unigram_part = 0
        if uni_denom != 0:
            unigram_part = LAMBDA_3 * uni_numer / uni_denom

        prob = trigram_part + bigram_part + unigram_part
        log_prob = math.log(prob, 2)
        return log_prob


    perplexity = n_gram_utils.eval_model('data/brown_dev.txt', model, get_log_prob)
    print(str(perplexity))


if __name__ == '__main__':
    main()
