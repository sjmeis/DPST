#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer, util

e1 =  SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cuda")
e2 =  SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
e3 =  SentenceTransformer("thenlper/gte-small", device="cuda")

def score(original, private):
    all_scores = []
    for model in [e1, e2, e3]:
        orig_embed = model.encode(original, convert_to_tensor=True, show_progress_bar=True)
        priv_embed = model.encode(private, convert_to_tensor=True, show_progress_bar=True)
        scores = util.pairwise_cos_sim(orig_embed, priv_embed).cpu()
        all_scores.append(float(scores.numpy().mean()))

    return round(float(np.mean(all_scores)), 3), all_scores

fields = {
    "reuters": "text",
    "spookyauthor": "text",
    "trustpilot": "text",
    "yelp": "review",
    "reddit-mental-health":"text"
}

orig = {}
for f in fields:
    orig[f] = pd.read_csv("data/datasets/{}.csv".format(f))

log = {}
for file in Path("data/").glob("*.csv"):
    print(file.stem)
    name = file.stem.split("_")[0]
    priv = pd.read_csv(file).dropna()
    x, y = score(orig[name].iloc[priv.index][fields[name]].to_list(), priv[fields[name]].to_list())
    log[file.stem] = {}
    log[file.stem]["CS"] = x
    log[file.stem]["scores"] = y
    print(x)

    with open("data/cs_log.json", 'w') as out:
        json.dump(log, out, indent=3)




