# class code for LLM-based Differential Privacy mechanisms

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList, pipeline
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartDecoder
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import mpmath
from mpmath import mp

import multiprocessing as mult

import nltk
nltk.download('punkt', quiet=True)

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

class ClipLogitsProcessor(LogitsProcessor):
  def __init__(self, min=-100, max=100):
    self.min = min
    self.max = max

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    scores = torch.clamp(scores, min=self.min, max=self.max)
    return scores
  
class DPPrompt():
    model_checkpoint = None
    min_logit = None
    max_logit = None
    sensitivity = None
    logits_processor = None

    tokenizer = None
    model = None
    device = None

    # flan-t5-xl: (-9.6643, 18.487299999999998)
    def __init__(self, model_checkpoint="google/flan-t5-large", min_logit=-11.8514, max_logit=14.6465, batch_size=16):
        self.model_checkpoint = model_checkpoint

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint).to(self.device)

        self.min_logit = min_logit
        self.max_logit = max_logit
        self.sensitivity = abs(self.max_logit - self.min_logit)
        self.logits_processor = LogitsProcessorList([ClipLogitsProcessor(self.min_logit, self.max_logit)])

        self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device, truncation=True)
        self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id

    def prompt_template_fn(self, doc):
        prompt = "Document : {}\nParaphrase of the document :".format(doc)
        return prompt
    
    def privatize(self, text, epsilon=100):
        prompt = self.prompt_template_fn(text)

        model_inputs = self.tokenizer(prompt, max_length=512, truncation=True, return_tensors="pt").to(self.device)

        per_token_epsilon = epsilon / len(model_inputs["input_ids"][0])
        temperature = 2 * self.sensitivity / per_token_epsilon

        output = self.model.generate(
            **model_inputs,
            do_sample = True,
            # top_k=0,
            # top_p=1.0,
            temperature=temperature,
            max_new_tokens=len(model_inputs["input_ids"][0]),
            logits_processor=self.logits_processor)

        private_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return private_text
    
    def privatize_dp(self, texts, epsilon=100, max_new_tokens=32):
        temperature = 2 * self.sensitivity / epsilon
        prompts = ListDataset(texts)
        private_texts = []
        for r in self.pipe(prompts, do_sample=True, top_k=0, top_p=1.0, temperature=temperature, logits_processor=self.logits_processor, max_new_tokens=max_new_tokens, batch_size=self.batch_size):
            private_texts.append(r[0]["generated_text"])
        return private_texts
    
class DPBart():
    model = None
    decoder = None
    tokenizer = None

    sigma = None
    num_sigmas = None
    c_min = None
    c_max = None
    delta = None

    def __init__(self, model='facebook/bart-large', num_sigmas=1/2):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = BartTokenizer.from_pretrained(model)
        self.model = BartModel.from_pretrained(model).to(self.device)
        self.decoder = BartForConditionalGeneration.from_pretrained(model).to(self.device)

        self.delta = 1e-5
        self.sigma = 0.2
        self.num_sigmas = num_sigmas
        self.c_min = -self.sigma
        self.c_max = self.num_sigmas * self.sigma

    def clip(self, vector):
        return torch.clip(vector, self.c_min, self.c_max)

    # https://github.com/trusthlt/dp-bart-private-rewriting/
    def calibrateAnalyticGaussianMechanism_precision(self, epsilon, delta, GS, tol = 1.e-12):
        """
        High-precision version of `calibrateAnalyticGaussianMechanism`.
        Should not be used for epsilon values above 5000.
        """

        if epsilon <= 1000:
            mp.dps = 500  # works for epsilon = 1000
        elif epsilon <= 2500 and epsilon > 1000:
            mp.dps = 1100  # works for epsilon = 2500
        else:
            mp.dps = 2200  # works for epsilon = 5000

        def Phi(t):
            return 0.5*(1.0 + mpmath.erf(t/mpmath.sqrt(2.0)))

        def caseA(epsilon,s):
            return Phi(mpmath.sqrt(epsilon*s)) - mpmath.exp(epsilon)*Phi(-mpmath.sqrt(epsilon*(s+2.0)))

        def caseB(epsilon,s):
            return Phi(-mpmath.sqrt(epsilon*s)) - mpmath.exp(epsilon)*Phi(-mpmath.sqrt(epsilon*(s+2.0)))

        def doubling_trick(predicate_stop, s_inf, s_sup):
            while(not predicate_stop(s_sup)):
                s_inf = s_sup
                s_sup = 2.0*s_inf
            return s_inf, s_sup

        def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
            s_mid = s_inf + (s_sup-s_inf)/2.0
            while(not predicate_stop(s_mid)):
                if (predicate_left(s_mid)):
                    s_sup = s_mid
                else:
                    s_inf = s_mid
                s_mid = s_inf + (s_sup-s_inf)/2.0
            return s_mid

        delta_thr = caseA(epsilon, 0.0)

        if (delta == delta_thr):
            alpha = 1.0

        else:
            if (delta > delta_thr):
                predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
                function_s_to_delta = lambda s : caseA(epsilon, s)
                predicate_left_BS = lambda s : function_s_to_delta(s) > delta
                function_s_to_alpha = lambda s : mpmath.sqrt(1.0 + s/2.0) - mpmath.sqrt(s/2.0)

            else:
                predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
                function_s_to_delta = lambda s : caseB(epsilon, s)
                predicate_left_BS = lambda s : function_s_to_delta(s) < delta
                function_s_to_alpha = lambda s : mpmath.sqrt(1.0 + s/2.0) + mpmath.sqrt(s/2.0)

            predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

            s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
            s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
            alpha = function_s_to_alpha(s_final)

        sigma = alpha*GS/mpmath.sqrt(2.0*epsilon)

        return float(sigma)
    
    def noise(self, vector, epsilon, delta=1e-5, method="gaussian"):
        k = vector.shape[-1]
        if method == "laplace":
            sensitivity = 2 * self.sigma * self.num_sigmas * k
            Z = torch.from_numpy(np.random.laplace(0, sensitivity / epsilon, size=k))
        elif method == 'gaussian':
            sensitivity = 2 * self.sigma * self.num_sigmas * np.sqrt(k)
            scale = np.sqrt((sensitivity**2 / epsilon**2) * 2 * np.log(1.25 / self.delta))
            Z = torch.from_numpy(np.random.normal(0, scale, size=k))
        elif method == "analytic_gaussian":
            sensitivity = 2 * self.sigma * self.num_sigmas * np.sqrt(k)
            analytic_scale = self.calibrateAnalyticGaussianMechanism_precision(epsilon, self.delta, sensitivity)
            Z = torch.from_numpy(np.random.normal(0, analytic_scale, size=k))
        return vector + Z

    def privatize(self, text, epsilon=100, method="analytic_gaussian"):
        inputs = self.tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(self.device)
        num_tokens = len(inputs["input_ids"][0])

        enc_output = self.model.encoder(**inputs)
        enc_output["last_hidden_state"] = self.noise(self.clip(enc_output["last_hidden_state"].cpu()), epsilon=epsilon, delta=self.delta, method=method).float().to(self.device)

        dec_out = self.decoder.generate(encoder_outputs=enc_output, max_new_tokens=num_tokens)
        private_text = self.tokenizer.decode(dec_out[0], skip_special_tokens=True)
        return private_text.strip()
    
    def privatize_batch(self, texts, epsilon=100, method="analytic_gaussian"):
        inputs = self.tokenizer(texts, max_length=512, truncation=True, padding=True, return_tensors="pt").to(self.device)
        num_tokens = [len(x) for x in inputs["input_ids"]]

        enc_output = self.model.encoder(**inputs)
        for i, x in tqdm(enumerate(enc_output["last_hidden_state"].cpu())):
            enc_output["last_hidden_state"][i] = self.noise(self.clip(x), epsilon=epsilon, delta=self.delta, method=method).float().to(self.device)

        dec_out = self.decoder.generate(encoder_outputs=enc_output, max_new_tokens=max(num_tokens))
        private_text = [self.tokenizer.decode(x, skip_special_tokens=True).strip() for x in dec_out]
        return [" ".join(x) for x in private_text]
