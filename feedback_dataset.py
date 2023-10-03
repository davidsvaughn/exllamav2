import os, sys, traceback, re
import numpy as np
import random
from random import randrange
from glob import glob
import ntpath
import json
import matplotlib.pyplot as plt
import subprocess

from itertools import chain
from functools import partial

from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset, Features, Value
# from datasets import load_dataset, DatasetDict, Dataset, Features, Value
# from datasets import concatenate_datasets, load_from_disk
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
# from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, get_scheduler
# from torch.optim.lr_scheduler import LinearLR, StepLR, MultiStepLR, LambdaLR, SequentialLR

txt,jpg = '.txt','.jpg'

##############################################################################
# # Load dataset from the hub
# ss = load_dataset("samsum")
# print(f"Train dataset size: {len(ss['train'])}")
# print(f"Test dataset size: {len(ss['test'])}")
# ss_train, ss_test = ss['train'], ss['test']
# # sys.exit()
##############################################################################

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self
        
def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data
    # return adict(data)

def read_lines(fn, sep='\n'):
    with open(fn, 'r') as f:
        lines = [n.strip() for n in f.readlines()]
    return lines
        
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filenames(path, ext='.txt', noext=False):
    pattern = os.path.join(path, f'*{ext}')
    x = np.array([path_leaf(f) for f in glob(pattern)])
    if noext:
        x = np.array([f.replace(ext,'') for f in x])
    return x

def sentence_span(span, sidx, text, sbuf=[0,0], maxtok=-1):
    b1,b2 = sbuf
    while True:
        i = (sidx<=span[0]).sum()-1
        j = (sidx<span[1]).sum() if b2>-1 else i
        i = max(0, i-b1)
        j = max(i, min(len(sidx)-1, j+b2)) if b2>-1 else j
        s = text[sidx[i]:sidx[j]]
        if maxtok<0 or i==0 or b1==0:
            break
        if len(s.split()) <= maxtok:
            break
        b1 -= 1
    return s

instruct_short = "Please give feedback on the following excerpt from an essay:"
instruct_long = """\
A student is writing an essay.  Please give feedback on the following excerpt from the essay.  \
Don't be concerned with factual correctness.  Instead, focus on helping the student to improve her writing.  \
Point out a problem you noticed, and/or give a suggestion for how the student could fix the problem, \
but avoid actually rewriting the section yourself.  \
Please keep your response short and limit it to only one or two sentences at the most.
Here is the excerpt: \
    """
context_short = "Here is the excerpt shown in the context of the whole essay.  The excerpt is enclosed in square brackets:"
context_long = """\
Here is the excerpt shown in the larger context of the essay.  The excerpt under scrutiny has been \
enclosed in brackets.  Please focus your feedback on the excerpt only, and not on the essay \
as a whole.
Here is the context: \
    """
# UID = 0
def parse_feedback(data, sbuf, maxtok=-1, maxtok_output=80, file_name=None):
    # global UID
    text = data['text']
    sidx = np.array(data['sent_idx'])
    fid = file_name.split('.')[0]
    items = []
    for i,fb in enumerate(data['feedback']):
        fb = adict(fb)
        
        if fb.type>0: continue
        if len(fb.comment.split())>maxtok_output: continue
        
        section = sentence_span(fb.span, sidx, text).strip()
        n = len(section.split())
        
        context = sentence_span(fb.span, sidx, text, sbuf=sbuf, maxtok=maxtok-n).strip()
        # enclose section in brackets
        if section not in context:
            print(f'WARNING ({file_name}): section not found in context:\n\t{section}\n\t{context}')
            sys.exit()
        context = context.replace(section, f' [ {section} ] ')

        num_tok = 1.4*(n+len(context.split()))
        if maxtok>0 and num_tok>maxtok: continue
    
        # uid = UID = UID+1
        uid = f'{fid}.{i}'
        
        ##########################
        (txt1, txt2) = (instruct_long, context_long)
        # (txt1, txt2) = (instruct_short, context_short)

        ## prior and post context:
        essay = context if context else ''
        instruction = f"""
### Instruction\n{txt1}\n{section} \
"""
        context = None if not essay else f"""
### Context\n{txt2}\n{essay} \
"""
        response = f"""
### Answer\n{fb.comment} \
"""
        prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])

        num_tok = 1.4*len(prompt.split())
        if maxtok>0 and num_tok>maxtok:
            # print(num_tok)
            continue
        
        item = adict( {'id':uid,
                        'prompt': prompt    
                        } )

        items.append(item)
    return items

def unzip_data(dataset):
    idx, prompts = [],[]
    for item in dataset:
        idx.append(item.id)
        prompts.append(item.prompt)
    return adict({'idx':idx, 'prompts':prompts })

def load_feedback_data(data_path, sbuf=[20,-1], maxtok=600, split=0.98, seed=1234, debug=False, test_frac=None, shuffle=True):
    json_files = get_filenames(data_path, ext='.json')
    random.seed(seed)
    train_data, test_data, N = [],[],[]
    for i,fn in enumerate(json_files):
        # print(f'{i}/{len(json_files)} :\t{fn}')
        # if '61727_11' in fn: print('HAH!')
        data = read_json(data_path + fn)
        items = parse_feedback(data, sbuf, maxtok=maxtok, file_name=fn)
        if random.random() < abs(split):
            train_data.extend(items)
        else:
            test_data.extend(items)
        if debug:
            for item in items:
                N.append(len(item.prompt.split()))

    # special case: split<0 means swap train and test
    if split<0:
        train_data, test_data = test_data, train_data

    # shuffle data
    if shuffle:
        random.shuffle(train_data)
        if test_frac is not None:
            random.shuffle(test_data)
            # take subset of test data
            n = int(len(test_data)*test_frac)
            test_data = test_data[:n]
        else:
            random.shuffle(test_data)

    # show token distribution
    if debug:
        N = np.array(N)
        print(len(N))
        plt.hist(N, 50)
        plt.title('input word counts')
        plt.show()
    
    # return
    return unzip_data(train_data), unzip_data(test_data)

def make_dataset(data):
    dataset = Dataset.from_dict({
        "id": data.idx,
        "prompt": data.prompts,
        # "feedback": data.labels
        },
        features=Features({
            "id": Value(dtype='string'), 
            "prompt": Value(dtype='string'),
            # "feedback": Value(dtype='string')
            }))
    return dataset

def get_mdl():
    line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
    line = line_as_bytes.decode("ascii")
    _, line = line.split(":", 1)
    line, _ = line.split("(")
    return line.strip()

def add_labels(sample):
    sample["labels"] = sample["input_ids"].copy()
    return sample

# empty list to save remainder from batches to use in next batch
remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

def chunk(sample, chunk_length=2048):
    # define global remainder variable to save remainder from batches to use in next batch
    global remainder
    # Concatenate all texts and add remainder from previous batch
    concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
    concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
    # get total number of tokens for batch
    batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

    # get max number of chunks for batch
    if batch_total_length >= chunk_length:
        batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
        for k, t in concatenated_examples.items()
    }
    # add remainder to global variable for next batch
    remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}

    # prepare labels
    result = add_labels(result)
    # result["labels"] = result["input_ids"].copy()

    return result

def format_sample(sample):
    return sample['prompt']

# tokenize and chunk dataset
def tok_chunk(dataset, tokenizer, max_seq_length=2048, packing=False):
    feats = list(dataset.features)
    # feats = ['text']

    if packing:
        tok_func = lambda sample: tokenizer(sample["text"])
        tokenized_dataset = dataset.map(tok_func, batched=True, remove_columns=feats)
    else:
        tok_func = lambda sample: tokenizer(sample["text"], truncation=True, max_length=max_seq_length, padding='max_length')
        tokenized_dataset = dataset.map(tok_func, batched=True, remove_columns=feats)

    input_lengths = np.array([len(x) for x in tokenized_dataset["input_ids"]])
    
    plt.hist(input_lengths, 50)
    plt.title('input token counts')
    plt.show()
    
    if packing:
        return tokenized_dataset.map(partial(chunk, chunk_length=max_seq_length), batched=True)
    
    return tokenized_dataset.map(add_labels, batched=True)

    # dataset = dataset.map(
    #     lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
    # ).map(
    #     partial(chunk, chunk_length=2048),
    #     batched=True,
    # )
    # return dataset


##############################################################################
# https://www.philschmid.de/sagemaker-llama2-qlora

def load_feedback_dataset(data_path = 'data', 
                          model_id = 'meta-llama/Llama-2-13b-hf',
                          split = 0.95,
                          sbuf = [8,8],
                          seed = 98765,
                          tokenizer = None,
                          test_frac=None,
                          tokenize=True,
                          max_seq_length=2048,
                          packing=False,
                          shuffle=True,
                          debug=False,
                          ):
    
    maxtok = max_seq_length
    # maxtok = 640

    # maxtok = 1024
    # sbuf = [20,-1]
    # sbuf = [5,5]
    # model_max_length = 1024
    # model_id = "meta-llama/Llama-2-13b-hf" # sharded weights
    # data_path = '/home/david/Documents/MeasInc/samples/feedback/essays'
    # data_path = 'data'
    
    json_path = os.path.join(data_path, 'json/')
    seed0 = random.randint(0,1000000)
    print(f'seed0 = {seed0}')

    train_data, test_data = load_feedback_data(json_path, sbuf, maxtok, split, seed, test_frac=test_frac, shuffle=shuffle)
    train_data = make_dataset(train_data)
    test_data = make_dataset(test_data)
    # dataset = DatasetDict({ 'train':train_data, 'test':test_data } )
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" 
    
    # template dataset to add prompt to each sample
    def template_dataset(sample):
        sample["text"] = f"{format_sample(sample)}{tokenizer.eos_token}"
        return sample
    
    # apply prompt template per sample
    # feats = list(train_data.features)
    feats = ['prompt']

    train_data = train_data.map(template_dataset, remove_columns=feats)
    test_data = test_data.map(template_dataset, remove_columns=feats)
    eval_data = test_data
    
    # print random sample
    random.seed(seed0)
    n = len(train_data)
    sample = train_data[randrange(len(train_data))]
    print(sample["text"])

    # if debug: sys.exit()
    
    # tokenize and chunk dataset
    if tokenize:
        train_data = tok_chunk(train_data, tokenizer, max_seq_length=max_seq_length, packing=packing)
        test_data = tok_chunk(test_data, tokenizer, max_seq_length=max_seq_length, packing=packing)

    # Print total number of samples
    print(f"\nTotal number of train samples: {len(train_data)}")
    print(f"Total number of test samples: {len(test_data)}")
    
    dataset = DatasetDict({ 'train':train_data, 'test':test_data, 'eval':eval_data } )
    
    return dataset if split<1 else train_data

if __name__ == "__main__":
    # dataset = load_feedback_dataset(debug=True, split=-0.5, test_frac=0.2)
    # dataset = load_feedback_dataset(split=0.9, sbuf = [8,5], tokenize=True, packing=False, max_seq_length=1024)

    dataset = load_feedback_dataset(data_path='/home/david/code/davidsvaughn/LLM-utils/llama2/code/data', 
                                    split=1, 
                                    sbuf = [8,5], 
                                    tokenize=False, 
                                    max_seq_length=1024, 
                                    shuffle=False)

    for sample in dataset:
        uid = sample["id"]
        text = sample["text"]
        n = text.index('### Answer')
        prompt = text[:n+10].strip()
        feedback = text[n+10:].replace('</s>','').strip()

        # print(prompt)
        print()
    