====================================================================================================
====================================================================================================
exllamav2
====================================================================================================
====================================================================================================
https://github.com/turboderp/exllamav2
==============================================================
== LAMBDA  ==

LAMBDA_PEM  AWS_W2_PEM

export LIP=ubuntu@209.20.157.61
ssh -i $LAMBDA_PEM $LIP

------------------------------------------------
sudo apt update
# sudo apt upgrade -y
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

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export HUG_READ_TOKEN=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
huggingface-cli login --token $HUG_READ_TOKEN
export HUG_WRITE_TOKEN=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
huggingface-cli login --token $HUG_WRITE_TOKEN

python llama2_merge.py

bash quantize.sh 3.0


# upload model  [ https://huggingface.co/docs/huggingface_hub/guides/upload ]

mkdir qmodels/tmp-70b-2.7bpw
mv qmodels/llamav2-70b-2.7bpw/out_tensor qmodels/tmp-70b-2.7bpw
huggingface-cli upload davidsvaughn/llamav2-70b-2.7bpw qmodels/llamav2-70b-2.7bpw


# retrieve model (local)
rsync -azP -e "ssh -i $LAMBDA_PEM" $LIP:/home/ubuntu/exllamav2/qmodels/llamav2-70b-2.7bpw/*.json /home/david/code/davidsvaughn/LLM-utils/exllamav2/models/qmodels/llamav2-70b-2.7bpw


# test quantized model!

python gen_feedback.py -m /home/ubuntu/exllamav2/qmodels/llamav2-70b-3.0bpw -p "prompts/prompt3.txt" -tm 0.7 -tk 40 -tp 0.9 -n 5

python gen_feedback.py -m /home/ubuntu/exllamav2/qmodels/llamav2-70b-2.7bpw -p "prompts/prompt3.txt" -tm 0.7 -tk 40 -tp 0.9 -n 5


















=======================================
vast.ai
=======================================

ssh-keygen -t rsa
ssh-add; ssh-add -l
cat ~/.ssh/id_rsa.pub

ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCBQ1qHPekHnWWveLwOY25OTBtbrZFrWmHC+7h7lDtqrS/umDNHweybsJLO1Lr3LrbX1212GiqgCQvbBjiibgl2paQjJvHcgdhURdqeUKrnBZq2FUgBRc3z/SeFR1UY4woNNADGKwSNBUcJ7YD3hBfxLbDsRe6bHoFKZoqGtgNMl5YP47EGokM6GqoYgEyqmC27RbIxBmoFfgthFRuBJ4svbOlK/DGD1umECRMuwMziUS5Iruc+Mz9cUU2TJIAqkKFMUkivwI9Bw5BG9p3L4ETr3u4b1+M8IdS2/MEsoyRvau0EI3x9xVF4YRLRyf31M+W/Z8MlQB1/mq2XxiqC4abRCdruEcdFa54eChO1ENBzsOqmt4yIM4IX6L2UcjSlAvciANo7n97hJqxCMDzlhCfYa3duzMZ7ElwAaJTXLQuC48AhAhTUyvAVMvISQQTpytR0NdFwXsuvCEVDbMGjCmhxazguRt6a7QO8KW6qnmmnzt7fiDXrGAUxLzYsDI+OTT8= david@camden


ssh -p 50440 root@145.14.10.31 -L 8080:localhost:8080

export LD_LIBRARY_PATH=/usr/local/cuda-12.0/targets/x86_64-linux/lib
