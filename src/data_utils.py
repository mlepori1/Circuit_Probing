#####
# From Amnesic Probing Repository
####
from tqdm import tqdm
import numpy as np

def read_conll_format(input_file):
    data = []
    with open(input_file, 'r') as f:
        sen = []
        tag = []
        pos = []
        dep = []
        orig_vals = []
        for line in tqdm(f):
            if line.strip() == '':
                pos_next_word = pos[1:] + ['EOL']
                tag_next_word = tag[1:] + ['EOL']
                data.append({'text': sen,
                             'labels': {
                                 'tag': tag,
                                 'pos': pos,
                                 'dep': dep,
                                 'orig_vals': orig_vals,

                                 'pos_next_word': pos_next_word,
                                 'tag_next_word': tag_next_word
                             }
                             })
                sen = []
                tag = []
                pos = []
                dep = []
                orig_vals = []
                continue
            vals = line.split('\t')
            if len(vals) > 1:
                sen.append(vals[1])
                tag.append(vals[3])
                pos.append(vals[4])
                dep.append(vals[7])
                orig_vals.append(vals)

    return data


def read_onto_notes_format(input_file):
    data = []
    for cur_file in tqdm(glob.glob(input_file + '/data/english/**/*.*gold_conll', recursive=True)):

        with open(cur_file, 'r') as in_f:
            sen = []
            ner = []
            np_start = []
            np_end = []
            phrase_start = []
            phrase_end = []
            prev_ner = ''
            for line in in_f:
                if line.startswith('#'):
                    continue
                if line.strip() == '':
                    data.append({'text': sen,
                                 'labels': {
                                     'ner': ner,
                                     'phrase_start': phrase_start,
                                     'phrase_end': phrase_end,
                                     'np_start': np_start,
                                     'np_end': np_end,
                                 }
                                 })
                    sen = []
                    ner = []
                    np_start = []
                    np_end = []
                    phrase_start = []
                    phrase_end = []
                    continue
                vals = line.split()
                sen.append(vals[3])

                cur_ner = vals[10]
                if cur_ner.startswith('('):
                    cur_ner = cur_ner[1:]
                    prev_ner = cur_ner
                if cur_ner.endswith(')'):
                    cur_ner = prev_ner[:-1]
                    prev_ner = ''
                if prev_ner != '':
                    cur_ner = prev_ner
                if cur_ner != '*' and cur_ner.endswith('*'):
                    cur_ner = cur_ner[:-1]
                ner.append(cur_ner)

                constituency = vals[5]

                if '(NP' in constituency:
                    np_start.append('S')
                else:
                    np_start.append('NS')

                if 'NP)' in constituency:
                    np_end.append('E')
                else:
                    np_end.append('NE')

                if constituency.startswith('('):
                    phrase_start.append('S')
                else:
                    phrase_start.append('NS')

                if constituency.endswith(')'):
                    phrase_end.append('E')
                else:
                    phrase_end.append('NE')

    return data

if __name__=="__main__":
    
    data = read_conll_format("../data/ud-treebanks-v2.12/UD_English-EWT/en_ewt-ud-train.conllu")
    print(data[0])

