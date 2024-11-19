import string
import json
import pandas as pd
import ast
import torch
from torch.utils.data import Dataset, DataLoader

def remove_punct(text):
    """removes some of the punctuation"""

    # from nltk import word_tokenize()
    # import string
    # punctuations = ['!', ',', '.', ':', ';']
    punctuations = list(string.punctuation)
    punctuations.remove(",")
    punctuations.remove(".")
    # punctuations.append("''") for Bert
    # punctuations.append("``") for Bert
    punctuations.append("«")
    punctuations.append("»")
    # punctuations.append("…") doesn't matter
    # punctuations.append("—") for Bert
    result = str()
    for char in text:
        if char == '-':
            # result+=' ' for Bert
            result+='-'
        if char not in punctuations:
            result+=char
    return result

def rp(text):
    """remove_punct() + strip()"""

    # return f' {text}'
    return f' {remove_punct(text).strip()}'

def find_tuple(X, Y):
    """find tuple X in tuple Y;
    is used only inside maps2tuples function"""

    # courtesy Claude 3.5 Sonnet

    len_x = len(X)
    len_y = len(Y)
    
    # If X is longer than Y, it can't be found in Y
    if len_x > len_y:
        return -1
    
    # Check each possible starting position in Y
    for i in range(len_y - len_x + 1):
        # Check if the slice of Y matches X
        if Y[i:i+len_x] == X:
            return i
            
    # If X is not found in Y, return -1
    return -1

def parse_data(jsonl_filepath):
    """ Parse competition data.
    """
    def __iter_parsed_json_lines(filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            for line in infile.readlines():
                yield json.loads(line)

    # Return list of parsed lines, presented in a form of dictionaries.
    return list(__iter_parsed_json_lines(filepath=jsonl_filepath))

class CustomDataset(Dataset):
    def __init__(
        self,
        path
    ):
        self.data = parse_data(path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx] 

class Collator():
    """convert datapoints to graph edge maps"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def collate(self, batch):
        """takes a batch, returns sent_ids,
        token_ids (model input data)
        and maps (model targets)"""

        inputs = self.tokenizer([rp(x['text']) for x in batch], padding = True, return_tensors = 'pt')
        ids = [x['sent_id'] for x in batch]
        # inputs = adaptive_pad(inputs, tokenizers)

        seq_len = inputs['input_ids'].size(-1)
        in_maps = torch.stack([self.in_map(x, seq_len) for x in batch], dim = 0)
        out_maps = torch.stack([self.out_map(x, seq_len) for x in batch], dim = 0)
        return ids, inputs, torch.stack((in_maps, out_maps), dim = 1)
    
    def tokenize(self, text, return_ids = False):
        """returns token or input_ids;
        decode() is needed if Roberta is used"""

        tokens = self.tokenizer(text)['input_ids']
        if return_ids:
            return tokens
        return [self.tokenizer.decode(x) for x in tokens]
    
    def chunk_tokenize(self, text, return_ids = False):
        """tokenizer for text chunks from opinions;
        this separate tokenizer is needed if Roberta is used"""

        # text = ' '+text
        tokens = self.tokenizer(text)['input_ids'][1:-1]
        if return_ids:
            return tokens
        return [self.tokenizer.decode(x) for x in tokens]
    
    def in_map(self, data, size):
        """takes datapoint, returns in_map,
        i.e. map of in_edges / spans;
        map size has to be the same for all batch elements"""

        feature_map = torch.zeros((size, size))
        if len(data['opinions']) == 0:
            return feature_map
        # from nltk import word_tokenize
        tokens = self.tokenize(rp(data['text']))
        # t_dict = dict(zip(tokens, list(range(len(tokens)))))
        for elem in data['opinions']:
            exp = self.chunk_tokenize(rp(elem['Polar_expression'][0][0]))
            exp_start = find_tuple(exp, tokens)
            if exp_start < 1:
                return feature_map
            feature_map[exp_start][exp_start:exp_start+len(exp)] = 1

            if elem['Polarity'] == 'NEG':
                feature_map[0][exp_start] = 1
            if elem['Polarity'] == 'POS':
                feature_map[exp_start][0] = 1

            if len(elem['Target'][0]) > 0:
                tg = self.chunk_tokenize(rp(elem['Target'][0][0]))
                tg_start = find_tuple(tg, tokens)
                if tg_start < 1:
                    return feature_map
                feature_map[tg_start][tg_start:tg_start+len(tg)] = 1
            if len(elem['Source'][0]) > 0:
                if 'NULL' in elem['Source'][0] or 'AUTHOR' in elem['Source'][0]:
                    continue
                src = self.chunk_tokenize(rp(elem['Source'][0][0]))
                src_start = find_tuple(src, tokens)
                if src_start < 1:
                    return feature_map
                feature_map[src_start][src_start:src_start+len(src)] = 1
        return feature_map

    def out_map(self, data, size):
        """takes datapoint, returns out_map,
        i.e. map of out_edges between expressions;
        map size has to be the same for all batch elements"""

        feature_map = torch.zeros((size, size))
        if len(data['opinions']) == 0:
            return feature_map
        # from nltk import word_tokenize
        tokens = self.tokenize(rp(data['text']))
        # print(f'{tokens=}')
        for elem in data['opinions']:
            exp = self.chunk_tokenize(rp(elem['Polar_expression'][0][0]))
            exp_mark = find_tuple(exp, tokens)
            # print(f'{exp_mark=}')
            if exp_mark < 1:
                return feature_map
            feature_map[0][exp_mark] = 1
            if len(elem['Target'][0]) > 0:
                tg = self.chunk_tokenize(rp(elem['Target'][0][0]))
                tg_mark = find_tuple(tg, tokens)
                # print(f'{tg=}; {tg_mark=}')
                if tg_mark < 1:
                    return feature_map
                feature_map[exp_mark][tg_mark] = 1
            if len(elem['Source'][0]) > 0:
                if 'NULL' in elem['Source'][0]:
                    continue
                elif 'AUTHOR' in elem['Source'][0]:
                    feature_map[tg_mark][0] = 1
                else:
                    src = self.chunk_tokenize(rp(elem['Source'][0][0]))
                    src_mark = find_tuple(src, tokens)
                    # print(f'{src_mark=}')
                    if src_mark < 1:
                        return feature_map
                    feature_map[tg_mark][src_mark] = 1
                    feature_map[src_mark][exp_mark] = 1
        return feature_map
    
    def maps2tuple(self, text, maps):
        """takes (text, (in_map, out_map))
        returns tuples (a list of lists)"""

        in_map, out_map = maps
        tokens = self.tokenize(rp(text))
        #t_dict = dict(zip(tokens, list(range(1,len(tokens)+1))))
        t_dict = dict(enumerate(tokens))
        tuples = list()
        start = (out_map[0] == 1).nonzero().flatten().tolist()
        if len(start) == 0:
            return []
        for exp in start:
            # print(exp, t_dict[exp])
            exp_span = extract_span(in_map, exp)
            exp_span = self.tokenizer.convert_tokens_to_string([t_dict[x] for x in exp_span])
            # exp_span = self.tokenizer.decode([t_dict[x] for x in exp_span])
            if in_map[0][exp] == 1:
                polarity = 'NEG'
            elif in_map[exp][0] == 1:
                polarity = 'POS'
            else:
                polarity = 'NEG'
            targets = (out_map[exp] == 1).nonzero().flatten().tolist()
            if len(targets) == 0:
                tuples.append(['NULL', [], exp_span[1:], polarity])
                continue
            for target in targets:
                tg_span = extract_span(in_map, target)
                tg_span = self.tokenizer.convert_tokens_to_string([t_dict[x] for x in tg_span])
                # tg_span = self.tokenizer.decode([t_dict[x] for x in tg_span])
                sources = (out_map[target] == 1).nonzero().flatten().tolist()
                if len(sources) == 0:
                    tuples.append(['NULL', tg_span[1:], exp_span[1:], polarity])
                else:
                    res = list()
                    for source in sources:
                        if source != 0:
                            exps = (out_map[source] == 1).nonzero().flatten().tolist()
                            if exp in exps:
                                res.append(source)
                    if len(res) == 1:
                        src_span = extract_span(in_map, res[0])
                        src_span = self.tokenizer.convert_tokens_to_string([t_dict[x] for x in src_span])
                        # src_span = self.tokenizer.decode([t_dict[x] for x in src_span])
                        tuples.append([src_span[1:], tg_span[1:], exp_span[1:], polarity])
                    elif len(res) == 0:
                        if 0 in sources:
                            tuples.append(['AUTHOR', tg_span[1:], exp_span[1:], polarity])
                        else:
                            tuples.append(['NULL', tg_span[1:], exp_span[1:], polarity])
                    else:
                        # print('ambiguous path')
                        pass
        return tuples
    
    def logits2preds(self, feature_map, threshold = 0.5):
        sigma = nn.Sigmoid()
        return (feature_map > threshold)

def extract_span(feature_map, index):
    """is used only inside maps2tuples function"""

    i, span = index, list()
    while feature_map[index][i] == 1:
        span.append(i)
        i+=1
    return span

def make_preds(model, dataset, col, threshold = 0.1):
    """takes model, dataset and collator
    and returns dataframe with predictions"""
    result = list()
    val_loader = DataLoader(dataset, batch_size = 1, shuffle = False,
                            collate_fn = col.collate)
    for ids, features, labels in tqdm(val_loader):
        text = dataset[len(result)]['text']
        preds = model(features)
        try:
            tuples = col.maps2tuple(text, logits2preds(preds[0], threshold))
            result.append([ids[0], text, tuples])
        except IndexError:
            result.append([ids[0], text, []])
    output = pd.DataFrame([(x[0], x[1], str2list(extract_tuple(x[2]))) for x in result],
                      columns = ['sent_id', 'text', 'pred'])
    return output

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

def plot_results(preds, targets, f_s = 6, cmap = 'Greys'):
    """takes two tensors of size (X, X)
    plots graph edge maps"""

    plt.rcParams.update({'font.size': 9})
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(f_s, f_s))
    axs[0, 0].imshow(preds[1], cmap=cmap, interpolation='none')
    axs[0, 0].title.set_text('Pred out_map')
    axs[0, 1].imshow(preds[0], cmap=cmap, interpolation='none')
    axs[0, 1].title.set_text('Pred in_map')
    axs[1, 0].imshow(targets[1], cmap=cmap, interpolation='none')
    axs[1, 0].title.set_text('Target out_map')
    axs[1, 1].imshow(targets[0], cmap=cmap, interpolation='none')
    axs[1, 1].title.set_text('Target in_map')
    
def drop_nan(df, tag):
    return df[~df[tag].isna()].loc[:, tag]

def visualize(log_dir):
    """takes log_dir, plots metrics;
    the following metrics are expected:
    plot 1:
    loss/train
    loss/val
    plot 2:
    BA_005/train
    BA_005/val
    BA_01/train
    BA_01/val
    BA_02/train
    BA_02/val
    plot 3:
    MAE/train
    MAE/val
    plot 4:
    MSE/train
    MSE/val"""

    reader = SummaryReader(log_dir, pivot=True)
    df = reader.scalars
    fig, axs = plt.subplots(2, 2, figsize=(16, 6))
    axs[0, 0].plot(drop_nan(df, "loss/train"), label="train")
    axs[0, 0].plot(drop_nan(df, "loss/val"), label="val")
    axs[0, 0].title.set_text('Loss')
    axs[0, 0].legend(loc="upper right")
    axs[0, 1].plot(drop_nan(df, "BA_005/train"), label="0.05 train")
    axs[0, 1].plot(drop_nan(df, "BA_005/val"), label="0.05 val")
    axs[0, 1].plot(drop_nan(df, "BA_01/train"), label="0.1 train")
    axs[0, 1].plot(drop_nan(df, "BA_01/val"), label="0.1 val")
    axs[0, 1].plot(drop_nan(df, "BA_02/train"), label="0.2 train")
    axs[0, 1].plot(drop_nan(df, "BA_02/val"), label="0.2 val")
    axs[0, 1].title.set_text('Binary accuracy')
    axs[0, 1].legend(loc="lower right")
    axs[1, 0].plot(drop_nan(df, "MAE/train"), label="train")
    axs[1, 0].plot(drop_nan(df, "MAE/val"), label="val")
    axs[1, 0].title.set_text('MAE')
    axs[1, 0].legend(loc="upper right")
    axs[1, 1].plot(drop_nan(df, "MSE/train"), label="train")
    axs[1, 1].plot(drop_nan(df, "MSE/val"), label="val")
    axs[1, 1].title.set_text('MSE')
    axs[1, 1].legend(loc="upper right")