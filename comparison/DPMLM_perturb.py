#!/usr/bin/env python3

import sys
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import string
import nltk
from transformers import AutoTokenizer

sys.path.insert(0, "/path/to/DPMLM/")

import DPMLM

punct = set(string.punctuation)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

fields = {
    "reuters": "text",
    "spookyauthor": "text",
    "trustpilot": "text",
    "yelp": "review",
    "reddit-mental-health":"text",
    "imdb10k":"text",
    "docnli":"premise"
}

words = {'reuters': 575.2059760956175, 'spookyauthor': 30.3869962715154, 'trustpilot': 59.74716853170566, 'yelp': 208.61572708875397, 'reddit-mental-health': 141.69536147095695, 'imdb10k': 265.5621, 'docnli': 338.1832019272887}

epsilons = {}
for w in words:
    mult = int(words[w])
    epsilons[w] = [mult*0.1, mult*0.5, mult*1] 

X = DPMLM.DPMLM()

for d in fields:
    data = pd.read_csv("data/datasets/{}.csv".format(d))
    for e in epsilons[d]:
        save_name = "data/{}_{}_{}.csv".format(d, "DPMLM", e)
        if Path(save_name).is_file() == True:
            continue
        
        print("{} | {}".format(d, e))
        temp = data.copy()
        private = []
        for x in tqdm(temp[fields[d]].to_list()):
            new_x = tokenizer.decode(tokenizer(x).input_ids[:256], skip_special_tokens=True)
            words = nltk.word_tokenize(new_x)
            per_token_eps = e / len([w for w in words if w not in punct])
            private.append(X.dpmlm_rewrite(new_x, epsilon=per_token_eps))
        temp[fields[d]] = private
        temp.to_csv(save_name, index=False)