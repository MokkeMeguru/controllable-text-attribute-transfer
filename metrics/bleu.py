"""calculate bleu score
"""
import math
import copy
from typing import List

def bleu_count(hypothesis: List[str], references: List[List[str]], max_n: int=4):
    """utility of bleu score calculation
    Args:
        hypothesis: list of generated sentence
        references: list of list of reference sentence
        max_n     : "n"-gram's max
    Returns:
        ret_clip_count List[int]: sum of shared hypothesis' ngram with references per n
        ret_count      List[int]: sum of hypothesis' ngram per n
        ret_len_hyp          int: sum of the closest reference words count in references
        ret_len_ref          int: sum of hypothesis words count
    Notes:
        references was written by some people.
        e.g.
        hypothesis[0] = "hello"
        references[0] = ["hello!", "hi", "good morning"]
    """

    # sum of hypothesis words count
    ret_len_hyp = 0
    # sum of the closest reference words count in references
    ret_len_ref = 0

    # sum of shared hypothesis' ngram with references per n
    ret_clip_count = [0] * max_n
    # sum of hypothesis' ngram per n
    ret_count = [0] * max_n

    assert len(hypothesis) == len(references), \
        'not acceptable. hyp and ref is same size'

    for hyp, ref in zip(hypothesis, references):

        # split hypothesis sentence to hypothesis words
        hyp_words = hyp.split()

        # split reference sentence to reference words
        ref_words_list = [r.split() for r in ref]

        n_ref = len(ref)

        # set upper bound
        # closest_diff: min(len(hyp_words) - len(ref_words))
        closest_diff = 9999
        closest_length = 9999

        # process for refenreces
        # reference's ngram list
        ref_ngram = dict()
        for ref_words in ref_words_list:
            diff = abs(len(ref_words) - len(hyp_words))
            if diff < closest_diff:
                closest_diff = diff
                closest_length = len(ref_words)
            elif diff == closest_diff and len(ref_words) < closest_length:
                closest_length = len(ref_words)

            for n in range(max_n):
                sent_ngram = dict()
                for start_idx in range(len(ref_words) - n):
                    # e.g.
                    # ngram = "3 how are you"
                    ngram = "{}".format(n + 1) + \
                        ''.join([" {}".format(ref_words[start_idx + k])
                                 for k in range(n + 1)])
                    if ngram not in sent_ngram:
                        sent_ngram[ngram] = 0
                    sent_ngram[ngram] += 1

                for ngram in sent_ngram.keys():
                    if ngram not in ref_ngram or ref_ngram[ngram] < sent_ngram[ngram]:
                        ref_ngram[ngram] = sent_ngram[ngram]

        ret_len_hyp += len(hyp_words)
        ret_len_ref += closest_length

        # process for hypothesis
        for n in range(max_n):
            hyp_ngram = dict()
            for start_idx in range(len(hyp_words) - n):
                ngram = "{}".format(n + 1) + \
                    ''.join([" {}".format(hyp_words[start_idx + k])
                             for k in range(n + 1)])
                if ngram not in hyp_ngram:
                    hyp_ngram[ngram] = 0
                hyp_ngram[ngram] += 1

            for ngram in hyp_ngram.keys():
                if ngram in ref_ngram:
                    ret_clip_count[n] += min(ref_ngram[ngram], hyp_ngram[ngram])
                ret_count[n] += hyp_ngram[ngram]

    return ret_clip_count, ret_count, ret_len_hyp, ret_len_ref


def safe_log(x):
    if x == 0:
        return -9999999999.0
    elif x < 0:
        raise Exception("Value Error")
    return math.log(x)

def corpus_bleu(hypothesis: List[str], references: List[List[str]], max_n: int=4):
    """utility of bleu score calculation
    Args:
        hypothesis: list of generated sentence
        references: list of list of reference sentence
        max_n     : "n"-gram's max
    Returns:
        bleu_scores      List[float]: [average bleu, 1-gram bleu, 2-gram bleu, ...]
        brevity_penalty        float:
        word_ratio             float: num of hyp's words / num of one ref's words
                                      ref is selected from references
        total_len_hyp            int: num of hyp's words
        totla_len_ref            int: num of one ref's words
                                      ref is selected from references
    Notes:
        references was written by some people.
        e.g.
        hypothesis[0] = "hello"
        references[0] = ["hello!", "hi", "good morning"]
    """

    clip_count, count, total_len_hyp, total_len_ref = bleu_count(
        hypothesis, references, max_n)


    # calculate each bleu score
    bleu_scores = []
    for n in range(max_n):
        if count[n] > 0:
            bleu_scores.append(clip_count[n] / count[n])
        else:
            bleu_scores.append(0)

    # penalty for short hypothesis
    if total_len_hyp < total_len_ref:
        if total_len_hyp == 0:
            brevity_penalty = 0.0
        else:
            brevity_penalty = math.exp(1 - total_len_ref / total_len_hyp)
    else:
        brevity_penalty = 1.0

    # calculate average bleu score
    log_bleu = sum([safe_log(bleu_scores[n]) for n in range(max_n)])
    bleu = brevity_penalty * math.exp(log_bleu / float(max_n))

    word_ratio = total_len_hyp / total_len_ref

    return ([bleu] + bleu_scores,
            brevity_penalty,
            word_ratio,
            total_len_hyp,
            total_len_ref)
