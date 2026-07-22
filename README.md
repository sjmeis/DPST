<div align="center">

  # DP-ST

  [![PyPI version](https://img.shields.io/pypi/v/dpst.svg)](https://pypi.org/project/dpst/)
  [![GitHub stars](https://img.shields.io/github/stars/sjmeis/DPST.svg?style=social)](https://github.com/sjmeis/DPST/stargazers)
  [![License](https://img.shields.io/github/license/sjmeis/DPST.svg)](https://github.com/sjmeis/DPST/blob/main/LICENSE)

</div>

Code repository for the EMNLP 2025 paper: *Leveraging Semantic Triples for Private Document Generation with Local Differential Privacy Guarantees*

## Getting Started
### Installation
You can now install `DP-ST` directly as a Python package.

```bash
# Install the core runtime engine
pip install dpst

# Install the optional data-preparation extensions required for the initial setup
pip install "dpst[setup]"
```

**Note on Heavy Runtimes**: We highly recommend installing the correct flavor of PyTorch matching your hardware's CUDA capability before installing this package.

#### More installation notes and tips
We have done our best to configure `DP-ST` and its set environment variables such that it can run as smoothly as possible on all setups. However, `vLLM` is currently quite finicky, so there may be some holdups with your runtime.

To summarize what we did:
* **`VLLM_ENABLE_V1_MULTIPROCESSING="0"`**: Forces vLLM to run in-process. Under WSL2 (if applicable to you), spawned worker subprocesses fail the `UvaBuffer` check (`UVA is not available`).
* **`VLLM_USE_FLASHINFER_SAMPLER="0"`**: Disables FlashInfer sampling backend, avoiding runtime errors on newer GPU architectures (SM120/Blackwell).
* **`VLLM_ATTENTION_BACKEND="TORCH_SDPA"`**: Directs attention operations through PyTorch's native Scaled Dot-Product Attention (SDPA), bypassing standalone `flash-attn` build dependencies.

Please also make sure to pass `hf_token` to initizalization, if applicable to reconstruction model you have chosen.

For the default embedding model (`jina-embeddings-v3`), you may see warnings like `flash_attn is not installed. Using PyTorch native attention implementation.` While these are generally safe to ignore, we highly recommend installing `flash-attn` (already in the requirements), as we have noticed particularly with `jina-embeddings-v3` that not doing so leads to headaches and odd behavior.

Finally, you will likely have to run `(uv) pip uninstall torchcodec` following installation, as this library creates compatability issues.

### Automated Database & Cluster Setup
In order to run `DP-ST`, you must first run the *preparation* stage as described in the paper. This includes booting up your local vector database, extracting triples from a public text corpus, clustering them, and storing them locally.

Ensure your local Weaviate instance is running, then execute the following automated command in your terminal:

```bash
dpst setup
```

**Note:** This can take a very long time! We recommend you set it and forget it. Alternatively, you can tweak the `max_rows` parameter of `initialize_database` to use less texts for the database preparation.

The above replaces the legacy workflow of manually executing `Triple2DB.ipynb` and `triple_cluster.ipynb`. The command will automatically stream the FineWeb public corpus dataset, extract/embed triples into Weaviate, run the MiniBatchKMeans allocations (50k, 100k, and 200k), and write the resulting cluster assets straight into the package data directory. If you prefer a more tailored approach, we still recommend using the notebooks.

## Usage

Running `DP-ST` is simple once the automated setup has completed:

```python
from dpst import DPST

# Initialize the engine (specify mode: "50k", "100k", or "200k")
X = DPST(mode="50k", model_checkpoint=MODEL_NAME, hf_token=TOKEN)

# Privatize your text corpus
private_texts = X.privatize([TEXTS], epsilon=DOC_PRIVACY_BUDGET) # or epsilon=LIST_OF_EPSILONS (for epsilon sweeps)

# Cleanup all resources (recommended!)
X.cleanup()
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
