#!/usr/bin/env python3

import sys
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import string 

sys.path.insert(0, "/path/to/MLDP/")

import MLDP

punct = set(string.punctuation)
detok = TreebankWordDetokenizer()

fields = {
    "reuters": "text",
    "spookyauthor": "text",
    "trustpilot": "text",
    "yelp": "review",
    "reddit-mental-health":"text"
}

words = {'reuters': 575.2059760956175, 'spookyauthor': 30.3869962715154, 'trustpilot': 59.74716853170566, 'yelp': 208.61572708875397, 'reddit-mental-health': 141.69536147095695}

epsilons = {}
for w in words:
    mult = int(words[w])
    epsilons[w] = [mult*0.1, mult*0.5, mult*1] 

X = MLDP.TEM()

for d in fields:
    data = pd.read_csv("data/datasets/{}.csv".format(d))
    for e in epsilons[d]:
        save_name = "data/{}_{}_{}.csv".format(d, "TEM", e)
        if Path(save_name).is_file() == True:
            continue
        
        print("{} | {}".format(d, e))
        temp = data.copy()
        private = []
        for x in tqdm(temp[fields[d]].to_list()):
            words = nltk.word_tokenize(x)
            per_token_eps = e / len([x for x in words if x not in punct])
            priv = [X.replace_word(word.lower(), epsilon=per_token_eps) if word not in punct else word for word in words]
            private.append(detok.detokenize(priv))
        temp[fields[d]] = private
        temp.to_csv(save_name, index=False)