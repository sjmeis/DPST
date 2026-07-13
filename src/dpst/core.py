import os
import json
from collections import defaultdict
from functools import partial
import importlib.resources as pkg_resources

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from datasketch import MinHash, MinHashLSH
from nltk import ngrams

import weaviate
from weaviate.classes.query import MetadataQuery, Filter

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import util
from openie import StanfordOpenIE

from vllm import LLM, SamplingParams

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

class DPST:
    def __init__(self, mode: str, hf_token: str = None, model_checkpoint: str = "meta-llama/Llama-3.2-1B-Instruct"):
        """
        DPST Runtime Engine. Assumes that `dpst setup` has already been executed
        to spin up the Weaviate collections and generate the local data clusters.
        """
        print("Initializing DPST Runtime...", flush=True)

        mode_map = {
            "50k": "fiftyk",
            "100k": "hundredk",
            "200k": "twohundredk"
        }

        if mode not in mode_map:
            raise ValueError("Error: [MODE] must be one of [50k, 100k, 200k].")
        
        self.mode = mode_map[mode]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            data_package = pkg_resources.files("dpst.data")
            
            with open(data_package / f"{mode}.json", 'r') as f:
                self.centroids = torch.tensor(json.load(f)).to(self.device)

            with open(data_package / f"{mode}_counts.json", 'r') as f:
                self.cluster_counts = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(
                f"Cluster data files for mode '{mode}' were not found. "
                "Please run `dpst setup` in your terminal to initialize database assets before using the runtime."
            )

        self.client = weaviate.connect_to_local()
        try:
            self.collection = self.client.collections.get("Triples")
        except Exception:
            self.client.close()
            raise RuntimeError(
                "The Weaviate collection 'Triples' does not exist. "
                "Ensure your local Weaviate instance is running and you have executed `dpst setup`."
            )

        self.properties = {
            "openie.affinity_probability_cap": 2 / 3,
            "openie.triple.strict": False,
        }
        self.IEclient = StanfordOpenIE(properties=self.properties)

        self.model_checkpoint = model_checkpoint
        self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to(self.device)

        print(f"Loading vLLM Engine for {self.model_checkpoint}...", flush=True)
        self.llm = LLM(
            model=self.model_checkpoint,
            trust_remote_code=True,
            tensor_parallel_size=1,
            dtype="bfloat16"
        )
        self.tokenizer = self.llm.get_tokenizer()

        self.ppl_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
        self.ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2")

        print("Initialization Finished.", flush=True)   

    def cleanup(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()

    def exponential(self, candidates, epsilon, sensitivity=1):
        probabilities = [np.exp(epsilon * x[1] / (2 * sensitivity)) for x in candidates]
        probabilities = probabilities / np.linalg.norm(probabilities, ord=1)
        return np.random.choice([x[0] for x in candidates], 1, p=probabilities)[0]

    def query_db(self, vector, cluster):
        response = self.collection.query.near_vector(
            near_vector=vector, 
            limit=self.cluster_counts[cluster], 
            filters=Filter.by_property(self.mode).equal(cluster), 
            return_metadata=MetadataQuery(distance=True)
        )
        candidates = [(x.properties["text"], max(1 - x.metadata.distance, 0)) for x in response.objects]
        return candidates

    def get_prompt(self, triples, messages=True):
        PROMPT = [
            {"role": "system", "content": "Generate a concise text for the given set of triples. Ensure that the generated output only includes the provided information from the triples, but feel free to fill in the gaps where sensible. If necessary, ignore triples that do not fit into the larger context. It is very important that the output is grammatically correct, natural, and logical. Provide a text that captures the semantic meaning of the triples, without being too verbose or lengthy. Do not provide any further explanation, only provide the output text."},
            {"role": "user", "content": "Input triples: [{’object’: ’Mike_Mularkey’,’property’: ’coach’,’subject’: ’Tennessee_Titans’}]"},
            {"role": "assistant", "content": "Output text: Mike Mularkey is the coach of the Tennessee Titans."},
            {"role": "user", "content": "Input triples: [{’object’: ’Albert_E._Austin’, ’property’: ’successor’, ’subject’: ’Alfred_N._Phillips’}, {’object’: ’Connecticut’, ’property’: ’birthPlace’, ’subject’: ’Alfred_N._Phillips’}, {’object’: ’United_States_House_of_Representatives’, ’property’: ’office’, ’subject’: ’Alfred_N._Phillips’}]"},
            {"role": "assistant", "content": "Output text: Albert E. Austin succeeded Alfred N. Phillips who was born in Connecticut and worked at the United States House of Representatives."},
            {"role": "user", "content": "Input triples: [{’object’: ’College_of_William_&_Mary’, ’property’: ’owner’, ’subject’: ’Alan_B._Miller_Hall’}, {’object’: ’2009-06-01’, ’property’: ’completionDate’, ’subject’: ’Alan_B._Miller_Hall’}, {’object’: ’101 Ukrop Way’, ’property’: ’address’, ’subject’: ’Alan_B._Miller_Hall’}, {’object’: ’Williamsburg,_Virginia’, ’property’: ’location’, ’subject’: ’Alan_B._Miller_Hall’}, {’object’: ’Robert_A._M._Stern’, ’property’: ’architect’, ’subject’: ’Alan_B._Miller_Hall’}]"},
            {"role": "assistant", "content": "Output text: The Alan B Miller Hall’s location is 101 Ukrop Way, Williamsburg, Virginia. It was designed by Robert A.M. Stern and was completed on 1 June 2009. Its owner is the College of William and Mary."},
            {"role": "user", "content": "Input Triples: {}".format(str(triples))}
        ]
        return PROMPT
    
    def compute_ppl(self, predictions, batch_size: int = 16, add_start_token: bool = True, max_length=32):
        if self.ppl_tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.ppl_tokenizer.special_tokens_map_extended.values())
            assert len(existing_special_tokens) > 0, "Model must have at least one special token to use for padding."
            self.ppl_tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        max_tokenized_len = max_length - 1 if (add_start_token and max_length) else max_length

        encodings = self.ppl_tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(torch.ge(attn_masks.sum(1), 2)), "Each input text must be at least two tokens long."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in range(0, len(encoded_texts), batch_size):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[self.ppl_tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(self.device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat([torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask], dim=1)

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.ppl_model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )
            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

    def get_triples_ie(self, text):
        res = [x for x in self.IEclient.annotate(text)]
        temp = [tuple(x.values()) for x in res]

        current = defaultdict(list)
        for t in temp:
            current[(t[0], t[1])].append(t)

        final = []
        for t in temp:
            s = "{} | {} | {}".format(t[0], t[1], t[2])
            if s not in final:
                final.append(s.replace("_", " "))
        
        lsh = MinHashLSH(threshold=0.4, num_perm=128)
        minhashes = {}
        for i, f in enumerate(final):
            minhash = MinHash(num_perm=128)
            for d in ngrams(f, 3):
                minhash.update("".join(d).encode('utf-8'))
            lsh.insert(i, minhash)
            minhashes[i] = minhash

        matches = {}
        for x, y in zip(final, minhashes):
            matches[x] = [final[z] for z in lsh.query(minhashes[y]) if z != y] 

        clusters = []
        covered = []
        for m in sorted(matches, key=lambda x: len(matches[x]), reverse=True):
            if m not in covered and len(matches[m]) > 0:
                clusters.append(matches[m])
                covered.extend(matches[m])

        clean = [x.replace(" | ", " ") for x in covered]
        if len(clean) == 0:
            return []
        ppls = dict(zip(covered, self.compute_ppl(predictions=clean, batch_size=64)["perplexities"]))

        best = []
        for c in clusters:
            scores = [ppls[x] for x in c]
            imin = np.argmin(scores)
            best.append(c[imin])

        ordered = []
        for f in final:
            if f in best:
                ordered.append(f)
        return ordered

    def privatize(self, texts, epsilon=10, DP=True):
        results = [None] * len(texts)
        prompts_to_generate = []
        indices_to_generate = []
        max_tokens_list = []

        for i, t in tqdm(enumerate(texts), total=len(texts), desc="Extracting and Perturbing Triples"):
            triples = self.get_triples_ie(t)
            if len(triples) == 0:
                results[i] = t
                continue

            if DP:
                eps = epsilon / len(triples)
                query_vectors = self.model.encode(triples, task="text-matching", truncate_dim=32, max_length=64)

                res = util.semantic_search(query_embeddings=torch.tensor(query_vectors).to(self.device), corpus_embeddings=self.centroids, top_k=1)
                clusters = [r[0]["corpus_id"] for r in res]

                candidates = []
                for q, c in zip(query_vectors, clusters):
                    near = self.query_db(q, c)
                    if len(near) > 0:
                        candidates.append(near)
                private_triples = [self.exponential(c, eps) for c in candidates]

                final = []
                for p in private_triples:
                    m = p.split(" | ")
                    final.append({"object": m[0], "property": m[1], "subject": m[2]})
                prompt = self.get_prompt(final)
            else:
                final = []
                for x in triples:
                    m = x.split(" | ")
                    final.append({"object": m[0], "property": m[1], "subject": m[2]})
                prompt = self.get_prompt(final)

            formatted_prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            
            prompts_to_generate.append(formatted_prompt)
            indices_to_generate.append(i)
            
            max_tokens = len(self.tokenizer.encode(t))
            max_tokens_list.append(max(max_tokens, 1))

        if prompts_to_generate:
            print(f"Batch executing generation for {len(prompts_to_generate)} documents...", flush=True)
            
            sampling_params_list = [
                SamplingParams(temperature=0.0, max_tokens=m_tok)
                for m_tok in max_tokens_list
            ]
            
            outputs = self.llm.generate(prompts_to_generate, sampling_params=sampling_params_list, use_tqdm=True)
            
            for idx, output in zip(indices_to_generate, outputs):
                generated = output.outputs[0].text

                generated = generated.split("Output text: ")[-1].strip().replace("\n", "")
                generated = generated.split("USER:")[0].strip().replace("\n", "")
                generated = generated.split("\t")[0].split("ASSISTANT")[0].split("USER")[0].split("###")[0].split("Note:")[0].split("Explanation:")[0].split("```")[0].split("EXPECTED_OUTPUT")[0]
                results[idx] = generated.strip()
                
        return results