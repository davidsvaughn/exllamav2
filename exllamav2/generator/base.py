from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer
)
from exllamav2.generator import (
    ExLlamaV2Sampler
)
import torch
import random

import torch.nn.functional as F

class ExLlamaV2BaseGenerator:

    # Internal state

    model: ExLlamaV2
    cache: ExLlamaV2Cache
    tokenizer: ExLlamaV2Tokenizer

    sequence_ids: torch.tensor = None

    def __init__(self, model, cache, tokenizer):

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer


    # For testing purposes, run a forward pass to make sure CUDA is fully initialized

    def warmup(self):

        input_ids = torch.zeros((1, 2), dtype = torch.long)
        self.model.forward(input_ids, cache = None, input_mask = None, preprocess_only = True)


    def full(self):

        return self.sequence_ids.shape[-1] >= self.model.config.max_seq_len


    def generate_simple(self, prompt: str or list, gen_settings: ExLlamaV2Sampler.Settings, num_tokens: int, seed = None):

        if seed is not None: random.seed(seed)

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        ids = self.tokenizer.encode(prompt)
        mask = self.tokenizer.padding_mask(ids) if batch_size > 1 else None

        overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0: ids = ids[:, -overflow:]

        self._gen_begin_base(ids, mask)

        for i in range(num_tokens):

            logits = self.model.forward(self.sequence_ids[:, -1:], self.cache, input_mask = mask).float().cpu()
            token, _ = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids, random.random())
            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

            if token == self.tokenizer.eos_token_id:
                break
            # if gen_settings.stop_tokens is not None and token in gen_settings.stop_tokens:
            #     break
            # if gen_settings.stop_sequence is not None and self.sequence_ids[:, -len(gen_settings.stop_sequence):].tolist() == gen_settings.stop_sequence:
            #     break

        num_prompt_tokens = ids.shape[-1]
        num_gen_tokens = self.sequence_ids.shape[-1] - num_prompt_tokens

        # text = self.tokenizer.decode(self.sequence_ids)
        text = self.tokenizer.decode(self.sequence_ids[:, num_prompt_tokens:])

        if isinstance(prompt, str): return text[0].strip(), num_gen_tokens

        return text.strip(), num_gen_tokens


    def _gen_begin_base(self, input_ids, mask = None):

        self.cache.current_seq_len = 0
        self.model.forward(input_ids[:, :-1], self.cache, input_mask = mask, preprocess_only = True)

        self.sequence_ids = input_ids.clone()
        self.sequence_ids = input_ids
    
    # experimental -------------------------------------------
    def generate_simple_samples(self, prompt: str or list, gen_settings: ExLlamaV2Sampler.Settings, num_tokens: int, num_samples=1, seed = None):
        if seed is not None: random.seed(seed)
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        ids = self.tokenizer.encode(prompt)
        mask = self.tokenizer.padding_mask(ids) if batch_size > 1 else None
        overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0: ids = ids[:, -overflow:]
        num_prompt_tokens = ids.shape[-1]

        self._gen_begin_base(ids, mask)

        responses, tot_toks = [],0
        for j in range(num_samples):

            for i in range(num_tokens):
                logits = self.model.forward(self.sequence_ids[:, -1:], self.cache, input_mask = mask).float().cpu()
                token, _ = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids, random.random())
                self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)
                if token == self.tokenizer.eos_token_id: break
                
            num_gen_tokens = self.sequence_ids.shape[-1] - num_prompt_tokens
            text = self.tokenizer.decode(self.sequence_ids[:, num_prompt_tokens:])
            tot_toks += num_gen_tokens
            responses.append(text[0].strip() if isinstance(prompt, str) else text.strip())
            
            self._gen_begin_base_exp(ids, mask)
        
        return responses, tot_toks


    def _gen_begin_base_exp(self, input_ids, mask = None):
        self.cache.current_seq_len = 0
        # self.model.forward(input_ids[:, :-1], self.cache, input_mask = mask, preprocess_only = True)
        self.sequence_ids = input_ids.clone()
        self.sequence_ids = input_ids