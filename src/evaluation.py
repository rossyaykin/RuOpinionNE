import sys
import os
import json

def tk(text):
    tokens = text.split()
    token_offsets = []
    i = 0
    for token in tokens:
        pos = text[i:].find(token)
        token_offsets.append((i + pos, i + pos + len(token)))
        i += pos + len(token)
    return token_offsets


def check_opinion_exist(htep, opinions_iter, check_diff_spans_valid_func):
    """ This function assess the new htep to be registered with respect to the
        task limitations on span values of `holder`, `target`, and `polarity`
    """

    exist = False

    # Unpack teh original tuple
    h, t, e, p = htep

    for o in opinions_iter:

        # Unpack the registered opinion
        h2, t2, e2, p2 = o

        is_matched = h == h2 and t == t2 and p == p2

        # Check whether `o` and given `htep` are matched.
        if not is_matched:
            continue

        # Extra check in the case when spans differs.
        if e != e2:
            check_diff_spans_valid_func(e, e2)
            continue

        # Otherwise it means that element exist.
        exist = True

    return exist


def convert_char_offsets_to_token_idxs(char_offsets, token_offsets):
    """
    char_offsets: list of str
    token_offsets: list of tuples

    >>> text = "I think the new uni ( ) is a great idea"
    >>> char_offsets = ["8:19"]
    >>> token_offsets =
    [(0,1), (2,7), (8,11), (12,15), (16,19), (20,21), (22,23), (24,26), (27,28), (29,34), (35,39)]

    >>> convert_char_offsets_to_token_idxs(char_offsets, token_offsets)
    >>> (2,3,4)
    """
    token_idxs = []

    for char_offset in char_offsets:
        bidx, eidx = char_offset.split(":")
        bidx, eidx = int(bidx), int(eidx)
        for i, (b, e) in enumerate(token_offsets):
            if b >= eidx or e <= bidx:
                intoken = False
            else:
                intoken = True
            if intoken:
                token_idxs.append(i)
    return frozenset(token_idxs)


def convert_opinion_to_tuple(sentence):
    text = sentence["text"]
    opinions = sentence["opinions"]
    opinion_tuples = []
    token_offsets = tk(text)

    if len(opinions) > 0:
        for opinion in opinions:

            # Extract idxs parts.
            holder_char_idxs = opinion["Source"][1]
            target_char_idxs = opinion["Target"][1]
            exp_char_idxs = opinion["Polar_expression"][1]

            # Compose elements of the new opinion.
            holder = frozenset(["AUTHOR"]) \
                if holder_char_idxs[0] == "NULL" \
                else convert_char_offsets_to_token_idxs(holder_char_idxs, token_offsets)
            target = convert_char_offsets_to_token_idxs(target_char_idxs, token_offsets)
            exp = convert_char_offsets_to_token_idxs(exp_char_idxs, token_offsets)
            polarity = opinion["Polarity"]

            assert polarity in ["POS", "NEG"], "wrong polarity mark: {}".format(sentence["sent_id"])

            htep = (holder, target, exp, polarity)

            def __check_diff_spans_valid_func(e1, e2):

                # There are no intersections.
                if len(e1.intersection(e2)) == 0:
                    return True

                # Intersections exist => raise an exception.
                raise Exception("expressions for the same holder, target and polarity "
                                "must not overlap: {}".format(sentence["sent_id"]))

            exist = check_opinion_exist(
                htep=htep,
                opinions_iter=iter(opinion_tuples),
                check_diff_spans_valid_func=__check_diff_spans_valid_func)

            if not exist:
                opinion_tuples.append(htep)

    return opinion_tuples


def sent_tuples_in_list(sent_tuple1, list_of_sent_tuples, keep_polarity=True):
    holder1, target1, exp1, pol1 = sent_tuple1
    if len(holder1) == 0:
        holder1 = frozenset(["_"])
    if len(target1) == 0:
        target1 = frozenset(["_"])
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        if len(holder2) == 0:
            holder2 = frozenset(["_"])
        if len(target2) == 0:
            target2 = frozenset(["_"])
        if (
            len(holder1.intersection(holder2)) > 0
            and len(target1.intersection(target2)) > 0
            and len(exp1.intersection(exp2)) > 0
        ):
            if keep_polarity:
                if pol1 == pol2:
                    return True
            else:
                return True
    return False

def weighted_score(sent_tuple1, list_of_sent_tuples):
    best_overlap = 0
    holder1, target1, exp1, pol1 = sent_tuple1
    if len(holder1) == 0:
        holder1 = frozenset(["_"])
    if len(target1) == 0:
        target1 = frozenset(["_"])
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        if len(holder2) == 0:
            holder2 = frozenset(["_"])
        if len(target2) == 0:
            target2 = frozenset(["_"])
        if (
            len(holder2.intersection(holder1)) > 0
            and len(target2.intersection(target1)) > 0
            and len(exp2.intersection(exp1)) > 0
        ):
            holder_overlap = len(holder2.intersection(holder1)) / len(holder1)
            target_overlap = len(target2.intersection(target1)) / len(target1)
            exp_overlap = len(exp2.intersection(exp1)) / len(exp1)
            overlap = (holder_overlap + target_overlap + exp_overlap) / 3
            if overlap > best_overlap:
                best_overlap = overlap
    return best_overlap


def tuple_precision(gold, pred, keep_polarity=True, weighted=True):
    """
    Weighted true positives / (true positives + false positives)
    """
    weighted_tp = []
    tp = []
    fp = []
    #
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]
        for stuple in ptuples:
            if sent_tuples_in_list(stuple, gtuples, keep_polarity):
                if weighted:
                    weighted_tp.append(weighted_score(stuple, gtuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fp.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fp) + 0.0000000000000001)


def tuple_recall(gold, pred, keep_polarity=True, weighted=True):
    """
    Weighted true positives / (true positives + false negatives)
    """
    weighted_tp = []
    tp = []
    fn = []
    #
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]
        for stuple in gtuples:
            if sent_tuples_in_list(stuple, ptuples, keep_polarity):
                if weighted:
                    weighted_tp.append(weighted_score(stuple, ptuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fn.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fn) + 0.0000000000000001)


def tuple_f1(gold, pred, keep_polarity=True, weighted=True):
    prec = tuple_precision(gold, pred, keep_polarity, weighted)
    rec = tuple_recall(gold, pred, keep_polarity, weighted)
    return 2 * (prec * rec) / (prec + rec + 0.00000000000000001)


def parse_data(jsonl_filepath):
    """ Parse competition data.
    """

    def __iter_parsed_json_lines(filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            for line in infile.readlines():
                yield json.loads(line)

    # Return list of parsed lines, presented in a form of dictionaries.
    return list(__iter_parsed_json_lines(filepath=jsonl_filepath))


def do_eval_core(gold, preds):
    """ Represent a core of the evaluation approach for
        the RuOpinionNE-2024 Competition.
    """
    assert(isinstance(gold, list))
    assert(isinstance(preds, list))

    # read in gold and predicted data, convert to dictionaries
    # where the sent_ids are keys
    check_gold = dict([(s["sent_id"], s['text']) for s in gold])
    check_preds = dict([(s["sent_id"], s['text']) for s in preds])

    g = set(check_gold.keys())
    p = set(check_preds.keys())

    assert g.issubset(p), "missing some sentences: {}".format(g.difference(p))
    assert p.issubset(g), "predictions contain sentences that are not in golds: {}".format(p.difference(g))
    for k in g:
        assert check_gold[k] == check_preds[k], "texts are not the same: {}".format(k)

    gold = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in gold])
    preds = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in preds])

    return tuple_f1(gold, preds)