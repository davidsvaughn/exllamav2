import torch
import os, sys
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# https://github.com/huggingface/peft/issues/692

# https://huggingface.co/TheBloke/guanaco-65B-GPTQ/discussions/2

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

model_size = 70 #  7  13  70

model_name = f'meta-llama/Llama-2-{model_size}b-hf'
adapter_name = f'davidsvaughn/llama-2-{model_size}b-feedback'
merged_model_name = f'davidsvaughn/llamav2-{model_size}b-merged'

###########################################
# load adapter model from hub
###########################################
step=1
print(f"\nStep {step}: load adapter model from hub"); step+=1

if '-7b-' in model_name:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )     
else: # 13b, 70b
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map={"": 0}
    )

print(f"\nStep {step}: load peft model"); step+=1
model = PeftModel.from_pretrained(model, adapter_name)

print(f"\nStep {step}: merge and unload"); step+=1
model = model.merge_and_unload()

print(f"\nStep {step}: load tokenizer from hub"); step+=1
tokenizer = AutoTokenizer.from_pretrained(adapter_name)

###########################################
# push merged model to HUB
###########################################

mkdirs(merged_model_name)

print(f"\nStep {step}: save model to {merged_model_name}"); step+=1
model.save_pretrained(
    merged_model_name,
    push_to_hub=True,
    private=True,
    safe_serialization=True,
    max_shard_size="10GB",
)

print(f"\nStep {step}: save tokenizer to {merged_model_name}"); step+=1
tokenizer.save_pretrained(
    merged_model_name,
    push_to_hub=True,
    private=True,
    repo_id=merged_model_name,
)

## --OR-- ??? ##
# tokenizer.push_to_hub(merged_model_name, private=True)
# model.push_to_hub(merged_model_name, private=True, safe_serialization=True)
# # model.push_to_hub(merged_model_name, private=True)