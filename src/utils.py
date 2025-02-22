import ast
import json
import pandas as pd
import sys
import os
import gc
import requests

def parse_data(jsonl_filepath):
    """ Parse competition data.
    """
    def __iter_parsed_json_lines(filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            for line in infile.readlines():
                yield json.loads(line)

    # Return list of parsed lines, presented in a form of dictionaries.
    return list(__iter_parsed_json_lines(filepath=jsonl_filepath))


def to_jsonl(data, target):
    """takes a list of dicts and path;
    saves the list to jsonl"""

    with open(target, "w") as f:
        for item in data:
            f.write(f"{json.dumps(item, ensure_ascii=False)}\n")


def save_jsonl(dataframe, path):
    """similar to to_jsonl but adds .jsonl to the path"""
    path+='.jsonl'
    return to_jsonl(dataframe, path)

def load_jsonl(url, path):
    """loads .jsonl from url,
    saves at path, reads it from path"""
    r = requests.get(url, allow_redirects=True)
    with open(path, 'wb') as st:
        st.write(r.content)
    return parse_data(path)

def dict2tuple(entry):
    """takes a single dataset entry (dict)
    and return a pair (text, opinion tuples)
    where opinion tuples is a list of lists"""

    if len(entry['opinions']) == 0:
        return (entry['text'], [])
    opinions = list()
    for elem in entry['opinions']:
        opinions.append([elem['Source'][0][0], elem['Target'][0][0], elem['Polar_expression'][0][0], elem['Polarity']])
    return (entry['text'], opinions)

def extract_tuple(text):
    """takes a text and tries to extract opinion tuples from it;
    returns opinion tuples as a list of lists;
    the text is supposed to be generated by a model"""

    if isinstance(text, list):
        return text
    lines = text.split('\n')
    for elem in lines:
        if len(elem.strip()) > 0 and elem.strip()[0] == '[' and ']' in elem:
            curr = elem.strip()
            while curr[-1] != ']':
                curr = curr[:-1]
            return curr
    return '[]'

def df2structure(df_pred):
    """takes a dataframe which has to contain the
    following columns: 'sent_id', 'text', 'pred';
    preds must be:
    (i) a list of lists OR
    (ii) a string which can be evaluated as a list of lists;
    returns a list of dicts
    which can be saved to .jsonl and submitted"""

    preds = list()
    for i, row in df_pred.iterrows():
        opinions = list()
        if isinstance(row['pred'], float):
            preds.append({'sent_id': row['sent_id'], 'text': row['text'], 'opinions': opinions})
            continue
        elif isinstance(row['pred'], str):
            curr = ast.literal_eval(row['pred'])
        else:
            curr = row['pred']
        for elem in curr:
            # holder
            try:
                h_ind = row['text'].index(elem[0])
                h_ind = f'{h_ind}:{h_ind+len(elem[0])}'
            except (ValueError, IndexError, TypeError):
                if elem[0] == 'NULL':
                    h_ind = '0:0'
                else:
                    h_ind = 'NULL'
            # target
            try:
                t_ind = row['text'].index(elem[1])
                t_ind = f'{t_ind}:{t_ind+len(elem[1])}'
            except (ValueError, IndexError, TypeError):
                t_ind = '0:0'
            # expression
            try:
                e_ind = row['text'].index(elem[2])
                e_ind = f'{e_ind}:{e_ind+len(elem[2])}'
            except (ValueError, IndexError, TypeError):
                e_ind = '0:0'
            # polarity
            try:
                if elem[3] in ('POS', 'NEG'):
                    pol = elem[3]
                else:
                    pol = 'NEG'
            except IndexError:
                pol = 'NEG'
            # append result
            if len(elem) == 4:
                opinions.append({'Source':[[elem[0]], [h_ind]], 'Target':[[elem[1]], [t_ind]],
                             'Polar_expression':[[elem[2]], [e_ind]], 'Polarity':pol})
            elif len(elem) == 3:
                opinions.append({'Source':[[elem[0]], [h_ind]], 'Target':[[elem[1]], [t_ind]],
                             'Polar_expression':[['0'], ['0:0']], 'Polarity':pol})
        result = {'sent_id': row['sent_id'], 'text': row['text'], 'opinions': opinions}
        try:
            convert_opinion_to_tuple(result) # check for errors
            preds.append(result)
        except Exception:
            preds.append({'sent_id': row['sent_id'], 'text': row['text'], 'opinions': []})
        # preds.append({'sent_id': row['sent_id'], 'text': row['text'], 'opinions': opinions})
    return preds

def str2list(text):
    """tries to evaluate a string as a list;
    doesn't change lists;
    returns None if fails"""

    if isinstance(text, list):
        return text
    try:
        res = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        res = None
    return res

def form_prompt(examples, text):
    """takes a list of example pairs (text, opinion tuples),
    where opinion tuples is a list of lists,
    and a single target text;
    returns a prompt with shots taken from example pairs
    and ending with a target text"""

    shots = '\n'.join([f'Текст: {pair[0]}\nОтвет: {pair[1]}' for pair in examples])
    return f"""Ты эксперт в оценке тональности.
Тебе нужно найти все негативные и позитивные отношения между сущностями в тексте и вывести их в следующем формате:
[источник отношения, объект отношения, выражение в тексте содержащее оценку, оценка (POS/NEG)]
Если источником отношения является автор, то пиши:
['AUTHOR', объект отношения, выражение в тексте содержащее оценку, оценка (POS/NEG)]
Если выраженного источника нет, то пиши:
['NULL', объект отношения, выражение в тексте содержащее оценку, оценка (POS/NEG)]
Допустимо вернуть пустой ответ:
[]
Не нужно давать пояснений к ответу.
Примеры
{shots}
Текст: {text}
Ответ: """

def short_report(dataframe):
    """takes a dataframe which has to contain the
    following columns: 'target', 'pred';
    counts and prints:
    (i) exact mathces between target and pred;
    (ii) accuracy;
    (iii) number of NaNs in pred
    does not return anything"""

    count = sum([row['target'] == row['pred'] for i, row in dataframe.iterrows()])
    print(f"Count: {count}")
    print(f"Accuracy: {count/len(dataframe):.3f}")
    print(f"NaNs: {dataframe.isna().sum()['pred']}")

# source
# https://github.com/dialogue-evaluation/RuOpinionNE-2024/blob/master/codalab/evaluation.py   

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