====================================================================================================
====================================================================================================
exllamav2
====================================================================================================
====================================================================================================
https://github.com/turboderp/exllamav2
==============================================================



--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
----------------------------------- QUANTIZE MODEL -----------------------------------
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------



LAMBDA_PEM  AWS_W2_PEM

export LIP=ubuntu@209.20.157.61
ssh -i $LAMBDA_PEM $LIP

------------------------------------------------
# install python3.9
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.9 -y
python3.9 --version

sudo apt install python3.9-dev -y
# sudo apt install python3-pip -y
------------------------------------------------

virtualenv -p python3.9 venv && source venv/bin/activate

virtualenv -p python3.8 venv38 && source venv38/bin/activate

git clone https://github.com/davidsvaughn/exllamav2 && cd exllamav2

pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118

# Successfully installed MarkupSafe-2.1.2 certifi-2022.12.7 charset-normalizer-2.1.1 filelock-3.9.0 fsspec-2023.4.0 idna-3.4 jinja2-3.1.2 mpmath-1.2.1 networkx-3.0rc1 numpy-1.24.1 pillow-9.3.0 pytorch-triton-2.1.0+6e4932cda8 requests-2.28.1 sympy-1.11.1 torch-2.2.0.dev20230919+cu118 torchvision-0.17.0.dev20230919+cu118 typing-extensions-4.4.0 

pip3 install -r requirements.txt

# Successfully installed accelerate-0.23.0 cramjam-2.7.0 fastparquet-2023.8.0 huggingface-hub-0.17.2 ninja-1.11.1 packaging-23.1 pandas-2.1.0 peft-0.4.0 psutil-5.9.5 python-dateutil-2.8.2 pytz-2023.3.post1 pyyaml-6.0.1 regex-2023.8.8 safetensors-0.3.3 sentencepiece-0.1.99 six-1.16.0 tokenizers-0.13.3 tqdm-4.66.1 transformers-4.33.2 tzdata-2023.3

# install flash attention?  2?
pip3 install flash-attn --no-build-isolation

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export HUG_READ_TOKEN=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
huggingface-cli login --token $HUG_READ_TOKEN
export HUG_WRITE_TOKEN=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
huggingface-cli login --token $HUG_WRITE_TOKEN

python llama2_merge.py

bash quantize.sh 3.0


# upload model  [ https://huggingface.co/docs/huggingface_hub/guides/upload ]

mkdir qmodels/tmp-70b-3.5bpw
mv qmodels/llamav2-70b-3.5bpw/out_tensor qmodels/tmp-70b-3.5bpw
huggingface-cli upload davidsvaughn/llamav2-70b-3.5bpw qmodels/llamav2-70b-3.5bpw


# retrieve model (local)
rsync -azP -e "ssh -i $LAMBDA_PEM" $LIP:/home/ubuntu/exllamav2/qmodels/llamav2-70b-2.7bpw/*.json /home/david/code/davidsvaughn/LLM-utils/exllamav2/models/qmodels/llamav2-70b-2.7bpw


# test quantized model!

python test_inference.py -m /home/ubuntu/exllamav2/qmodels/llamav2-70b-3.0bpw -p "prompts/prompt3.txt" -l 1024

python gen_feedback.py -m /home/ubuntu/exllamav2/qmodels/llamav2-70b-3.5bpw -p "prompts/prompt3.txt" -tm 0.7 -tk 50 -tp 0.95 -n 5 -l 1024

python gen_feedback.py -m /home/ubuntu/exllamav2/qmodels/llamav2-70b-3.0bpw -p "prompts/prompt3.txt" -tm 0.7 -tk 50 -tp 0.95 -n 5 -l 1024

python gen_feedback.py -m /home/ubuntu/exllamav2/qmodels/llamav2-70b-2.7bpw -p "prompts/prompt3.txt" -tm 0.7 -tk 50 -tp 0.95 -n 5 -l 1024

# max_seq_length ??  [ https://github.com/turboderp/exllamav2/issues/47 ]
-l 1024








--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
-------------------------------- RUN QUANTIZED MODEL ---------------------------------
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------


export LIP=ubuntu@209.20.157.61
ssh -i $LAMBDA_PEM $LIP

------------------------------------------------
# install python3.9
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.9 -y
python3.9 --version

sudo apt install python3.9-dev -y
# sudo apt install python3-pip -y
------------------------------------------------

virtualenv -p python3.9 venv && source venv/bin/activate
git clone https://github.com/davidsvaughn/exllamav2 && cd exllamav2
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
pip3 install -r requirements.txt

# install flash attention?
pip3 install flash-attn --no-build-isolation

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export HUG_READ_TOKEN=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
huggingface-cli login --token $HUG_READ_TOKEN

export BITS=4.0

git lfs install

git clone git@hf.co:davidsvaughn/llamav2-70b-"$BITS"bpw

python gen_feedback.py -m davidsvaughn/llamav2-70b-"$BITS"bpw -p "prompts/prompt3.txt" -tm 0.7 -tk 50 -tp 0.95 -n 5 -l 1024






================================================================================================================
================================================================================================================
================================================================================================================
RESAMPLING!!!!!!






============================================================================
Lambda
============================================================================

export LIP=ubuntu@209.20.159.239
ssh -i $LAMBDA_PEM $LIP

## copy data !!!
rsync -azP -e "ssh -i $LAMBDA_PEM" /home/david/code/davidsvaughn/LLM-utils/llama2/code/data $LIP:/home/ubuntu

## retrieve data !!!
rsync -azP -e "ssh -i $LAMBDA_PEM" $LIP:/home/ubuntu/exllamav2/llama*.txt /home/david/code/davidsvaughn/LLM-utils/exllamav2/samples/h100

------------------------------------------------
# install python3.9
sudo apt update && sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.9 -y
python3.9 --version

sudo apt install python3.9-dev -y
# sudo apt install python3-pip -y
------------------------------------------------

virtualenv -p python3.9 venv && source venv/bin/activate
git clone https://github.com/davidsvaughn/exllamav2 && cd exllamav2
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
pip3 install -r requirements.txt

# install flash attention?
pip3 install flash-attn --no-build-isolation

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export HUG_READ_TOKEN=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
huggingface-cli login --token $HUG_READ_TOKEN

sudo apt-get install git-lfs && git lfs install

# download model ####xxxx git clone git@hf.co:davidsvaughn/llamav2-70b-4.0bpw
git clone https://huggingface.co/davidsvaughn/llamav2-70b-4.0bpw
-> davidsvaughn@gmail.com
-> 5*gY8kpauf23.Wp

python sample_feedback.py -m llamav2-70b-4.0bpw -d "/home/ubuntu/data" -n 8 -l 1024 -I 0 -J 10000


============================================================================
vast.ai
============================================================================
ssh-keygen -t rsa
ssh-add; ssh-add -l
cat ~/.ssh/id_rsa.pub

ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCBQ1qHPekHnWWveLwOY25OTBtbrZFrWmHC+7h7lDtqrS/umDNHweybsJLO1Lr3LrbX1212GiqgCQvbBjiibgl2paQjJvHcgdhURdqeUKrnBZq2FUgBRc3z/SeFR1UY4woNNADGKwSNBUcJ7YD3hBfxLbDsRe6bHoFKZoqGtgNMl5YP47EGokM6GqoYgEyqmC27RbIxBmoFfgthFRuBJ4svbOlK/DGD1umECRMuwMziUS5Iruc+Mz9cUU2TJIAqkKFMUkivwI9Bw5BG9p3L4ETr3u4b1+M8IdS2/MEsoyRvau0EI3x9xVF4YRLRyf31M+W/Z8MlQB1/mq2XxiqC4abRCdruEcdFa54eChO1ENBzsOqmt4yIM4IX6L2UcjSlAvciANo7n97hJqxCMDzlhCfYa3duzMZ7ElwAaJTXLQuC48AhAhTUyvAVMvISQQTpytR0NdFwXsuvCEVDbMGjCmhxazguRt6a7QO8KW6qnmmnzt7fiDXrGAUxLzYsDI+OTT8= david@camden
------------------------------------------------------------------------------------
CLI

vastai set api-key f664b2d423657ae341a31e87ed114eba705ac0029c0dc02722525b47bb9d18b6

vastai search offers 'gpu_name=A100_PCIE rentable=any gpu_ram>35 disk_space>200'
vastai search offers 'num_gpus=1 gpu_ram>35 disk_space>200'
vastai create instance 7112000 --image nvidia/cuda:12.0.1-devel-ubuntu20.04 --disk 200

------------------------------------------------------------------------------------

ssh -p 11528 root@149.11.242.18 -L 8080:localhost:8080

ssh -p 40023 root@50.115.47.83 -L 8080:localhost:8080

## copy data !!!
rsync -azP -e "ssh -p 40023" /home/david/code/davidsvaughn/LLM-utils/llama2/code/data root@50.115.47.83:/root

## retrieve data !!!
rsync -azP -e "ssh -p 40023" root@50.115.47.83:/root/exllamav2/llama*.txt /home/david/code/davidsvaughn/LLM-utils/exllamav2/samples/a100

------------------------------------------------------------------------------------
Welcome to your vast.ai container! This session is running in `tmux`.
To disconnect without closing your processes, press ctrl+b, release, then d.
To disable auto-tmux, run `touch ~/.no_auto_tmux` and reconnect. See also https://tmuxcheatsheet.com/
------------------------------------------------------------------------------------

# install python3.9...
sudo apt update && sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.9 -y
python3.9 --version
sudo apt install python3.9-dev -y

# install pip3, virtualenv...
sudo apt install python3-pip -y
pip3 install virtualenv

# install packages...
git clone https://github.com/davidsvaughn/exllamav2 && cd exllamav2
virtualenv -p python3.9 venv && source venv/bin/activate

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip3 install -r requirements.txt
pip3 install flash-attn --no-build-isolation

sudo apt-get install git-lfs && git lfs install

export HUG_READ_TOKEN=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
huggingface-cli login --token $HUG_READ_TOKEN

# download model ####xxxx git clone git@hf.co:davidsvaughn/llamav2-70b-4.0bpw
git clone https://huggingface.co/davidsvaughn/llamav2-70b-4.0bpw
-> davidsvaughn@gmail.com
-> 5*gY8kpauf23.Wp


# run!!!
## python gen_feedback.py -m llamav2-70b-4.0bpw -p "prompts/prompt3.txt" -tm 0.7 -tk 50 -tp 0.95 -n 10 -l 1024

python sample_feedback.py -m llamav2-70b-4.0bpw -d "/root/data" -n 8 -l 1024 -I 10000 -J 20000

