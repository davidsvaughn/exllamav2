
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

from feedback_dataset import load_feedback_dataset

torch.cuda._lazy_init()
torch.set_printoptions(precision = 10)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.set_float32_matmul_precision("medium")

def append_lines(file_name, lines):
    txt = '\n'.join(lines) + '\n'
    with open(file_name, 'ab') as f:
        f.write(txt.encode("utf-8", "ignore"))

parser = argparse.ArgumentParser(description = "Get feedback with ExLlamaV2 model")
parser.add_argument("-m", "--model_dir", type = str, help = "Path to model directory")
parser.add_argument("-p", "--prompt", type = str, help = "Generate from prompt")
parser.add_argument("-d", "--data", type = str, default = '/root/data', help = "path to data")
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

parser.add_argument("-I", "--start", type = int, default = 0, help = "Start index")
parser.add_argument("-J", "--end", type = int, default = 100, help = "End index")

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


#-------------------------------------------------------
# Load dataset

dataset = load_feedback_dataset(data_path=args.data,
                                split=1, 
                                sbuf = [8,5], 
                                tokenize=False, 
                                max_seq_length=args.length, # 1024
                                shuffle=False)

I,J = args.start, args.end
output_file = f"llamav2-output-{I}-{J}.txt"
results_file = f"llamav2-results-{I}-{J}.txt"

# if os.path.exists(output_file):
#     os.path.remove(output_file)
# if os.path.exists(results_file):    
#     os.path.remove(results_file)

#-------------------------------------------------------
# prepare model

print("Loading model: " + args.model_dir)
model = ExLlamaV2(config)
model.load()

tokenizer = ExLlamaV2Tokenizer(config)
cache = ExLlamaV2Cache(model)
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
generator.warmup()

settings = ExLlamaV2Sampler.Settings()
settings.token_repetition_penalty = 1.15
settings.temperature = args.temp # 0.85
settings.top_k = args.top_k # 0.8
settings.top_p = args.top_p # 50

#-------------------------------------------------------

for i in range(I,J):
    if i >= len(dataset):
        break
    sample = dataset[i]
    uid = sample["id"]
    text = sample["text"]
    n = text.index('### Answer')
    prompt = text[:n+10].strip()
    feedback = text[n+10:].replace('</s>','').strip()

    print(f'{i}/{len(dataset)} : {uid}')

    out_lines = [f'{uid}\tgpt4\t{feedback}']
    res_lines = [f"\n{'==='* 50}\nINPUT PROMPT:\n{prompt}"]
    res_lines += [f"{'---'* 25}\nGPT-4 FEEDBACK:\n{feedback}"]

    # print('--------->')
    time_begin, t_minus, num_toks_total = time.time(), 0, 0

    #-------------------------------------------------------
    for i in range(args.num_samples):
        settings.temperature = random.uniform(0.4, 1)
        settings.top_p = random.uniform(0.85, 0.98)
        settings.top_k = random.randint(40, 60)

        response, num_gen_toks = generator.generate_simple(prompt, settings, max_new_tokens, seed=random.randint(0,1000000))
        t1 = time.time()
        # print(f"{i+1}.\t{response}")
        t_minus += (time.time()-t1)
        num_toks_total += num_gen_toks
        
        out_lines += [f'{uid}\tllama2\t{response}']
        res_lines += [f"{'---'* 25}\nLLAMA-2 FEEDBACK #{i+1}:\n{response}"]
    #-------------------------------------------------------

    time_total = time.time() - time_begin - t_minus
    # print('<---------\n')
    print(f"{args.num_samples} responses generated in {time_total:.2f} seconds, {num_toks_total} tokens, {num_toks_total / time_total:.2f} tokens/second")

    ## save output
    append_lines(output_file, out_lines)
    append_lines(results_file, res_lines)
