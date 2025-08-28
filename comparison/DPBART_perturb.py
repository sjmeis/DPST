#!/usr/bin/env python3

import sys
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path

import LLMDP

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


X = LLMDP.DPBart(model="bart-base")

for d in fields:
    data = pd.read_csv("data/datasets/{}.csv".format(d))
    for e in epsilons[d]:
        save_name = "data/{}_{}_{}.csv".format(d, "DPBART", e)
        if Path(save_name).is_file() == True:
            continue
        
        print("{} | {}".format(d, e))
        temp = data.copy()
        private = []
        for x in tqdm(temp[fields[d]].to_list()):
            private.append(X.privatize(x, epsilon=e))
        temp[fields[d]] = private
        temp.to_csv(save_name, index=False)