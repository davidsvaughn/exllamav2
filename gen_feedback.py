
from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    model_init,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

import argparse, os, math, time
import pandas, fastparquet
import torch
import torch.nn.functional as F
from conversion.tokenize import get_tokens
from conversion.quantize import list_live_tensors

import sys
import json
import random

torch.cuda._lazy_init()
torch.set_printoptions(precision = 10)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser(description = "Get feedback with ExLlamaV2 model")
parser.add_argument("-m", "--model_dir", type = str, help = "Path to model directory")
parser.add_argument("-p", "--prompt", type = str, help = "Generate from prompt")
parser.add_argument("-n", "--num_samples", type = int, default = 1, help = "number of samples")
parser.add_argument("-r", "--random_seed", type = int, default = -1, help = "random seed")

parser.add_argument("-tm", "--temp", type = float, default = 0.65, help = "temperature")
parser.add_argument("-tk", "--top_k", type = int, default = 50, help = "top_k")
parser.add_argument("-tp", "--top_p", type = float, default = 0.9, help = "top_p")

parser.add_argument("-l", "--length", type = int, help = "Maximum sequence length")
parser.add_argument("-rs", "--rope_scale", type = float, default = 1.0, help = "RoPE scaling factor")
parser.add_argument("-ra", "--rope_alpha", type = float, default = 1.0, help = "RoPE alpha value (NTK)")
parser.add_argument("-nfa", "--no_flash_attn", action = "store_true", help = "Disable Flash Attention")

parser.add_argument("-ed", "--eval_dataset", type = str, help = "Perplexity evaluation dataset (.parquet file)")
parser.add_argument("-er", "--eval_rows", type = int, default = 128, help = "Number of rows to apply from dataset")
parser.add_argument("-el", "--eval_length", type = int, default = 2048, help = "Max no. tokens per sample")
parser.add_argument("-t", "--tokens", type = int, default = 150, help = "Max no. tokens")
parser.add_argument("-ps", "--prompt_speed", action = "store_true", help = "Test prompt processing (batch) speed over context length")
parser.add_argument("-s", "--speed", action = "store_true", help = "Test raw generation speed over context length")

# Initialize model and tokenizer
args = parser.parse_args()

config = ExLlamaV2Config()
config.model_dir = args.model_dir
config.prepare()

# Set config options
if args.length: config.max_seq_len = args.length
config.rope_scale = args.rope_scale
config.rope_alpha = args.rope_alpha
config.no_flash_attn = args.no_flash_attn
max_new_tokens = args.tokens
seed = args.random_seed
if seed<0:
    seed = random.randint(0,1000000)
    print(f"random_seed = {seed}")
random.seed(seed)

# get prompt
prompt = args.prompt
# if prompt is a file, load text from file
if prompt.endswith('.txt') and os.path.exists(prompt):
    with open(prompt) as f:
        prompt = f.read()
prompt = prompt.strip()

print("Loading model: " + args.model_dir)
model = ExLlamaV2(config)
model.load()

tokenizer = ExLlamaV2Tokenizer(config)
cache = ExLlamaV2Cache(model)
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

# Generate some text
settings = ExLlamaV2Sampler.Settings()
settings.temperature = args.temp # 0.85
settings.top_k = args.top_k # 0.8
settings.top_p = args.top_p # 50
settings.token_repetition_penalty = 1.15
# settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
# settings.stop_tokens = [tokenizer.eos_token_id]
# settings.stop_sequence = tokenizer.encode('###')
# responses = []

print(prompt)
generator.warmup()

print('--------->')
time_begin, t_minus, num_toks_total = time.time(), 0, 0

#-------------------------------------------------------
for i in range(args.num_samples):
    response, num_gen_toks = generator.generate_simple(prompt, settings, max_new_tokens, seed=random.randint(0,1000000))
    t1 = time.time()
    print(f"{i+1}.\t{response}")
    t_minus += (time.time()-t1)
    num_toks_total += num_gen_toks
    # responses.append(response)
# for i,response in enumerate(responses): print(f"{i}.\t{response}")
#-------------------------------------------------------

time_total = time.time() - time_begin - t_minus
print('<---------\n')
print(f"Responses generated in {time_total:.2f} seconds, {num_toks_total} tokens, {num_toks_total / time_total:.2f} tokens/second")
