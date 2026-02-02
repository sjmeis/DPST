# DP-ST
Code repository for the EMNLP 2025 paper: *Leveraging Semantic Triples for Private Document Generation with Local Differential Privacy Guarantees*

## Getting Started
Please first make sure to install all requirements before proceeding:

```
pip install -r requirements.txt
```

In order to run `DP-ST`, you must first run the *preparation* stage as described in the paper. This includes extracting triples from a public text corpus, and then storing these locally into a vector database.

We include the `Triple2DB.ipynb` and `triple_cluster.ipynb` notebooks, which must be run (in order) to create the public triple corpus described in the paper. After this is complete, `DPST.py` can be imported and used. This includes saving the clusters from `triple_cluster.ipynb` to `data/clusters`.

Running `DP-ST` is simple if the setup was performed correctly:

```
import DPST
X = DPST.DPST(model_checkpoint=MODEL_NAME, hf_token=TOKEN)
private_texts = X.privatize([TEXTS], epsilon=DOC_PRIVACY_BUDGET)
```

`MODEL_NAME` refers to the model used for text reconstruction (i.e., the `Llama-3.2` models we use in the work), and `hf_token` is only necessary for gated models on Hugging Face.

## Running other DP Methods
In this repository (under the `comparison` directory), you will find a number of scripts (`*_perturb.py`) to reproduce the privatized texts as performed in our work.

The code for `DP-BART` and `DP-Prompt` can be found in `LLMDP.py`. `DP-MLM` can be found [here](https://github.com/sjmeis/DPMLM/) and the code for `TEM` can be found [here](https://github.com/sjmeis/MLDP/).

We also include the evaluation code for cosine similarity (`CS.py`) and G-Eval (`Geval.ipynb`), located in `evaluation`.

NOTE: in all provided notebooks, please make sure to include the correct libraries and link the paths accordingly. This is necessary for the code to run correctly!

## Citation
If you use this code in your research, please consider citing the published work:

```
@inproceedings{meisenbacher-etal-2025-leveraging,
    title = "Leveraging Semantic Triples for Private Document Generation with Local Differential Privacy Guarantees",
    author = "Meisenbacher, Stephen  and
      Chevli, Maulik  and
      Matthes, Florian",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.455/",
    doi = "10.18653/v1/2025.emnlp-main.455",
    pages = "8976--8992",
    ISBN = "979-8-89176-332-6"
}
```
