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

python llama2_save.py

bash quantize.sh

python test_inference.py -m /home/ubuntu/code/qmodels/llamav2-7b-3.0bpw -p "prompts/prompt3.txt"

rsync -azP -e "ssh -i $LAMBDA_PEM" $LIP:/home/ubuntu/exllamav2/qmodels /home/david/code/davidsvaughn/LLM-utils/exllamav2/models



=======================================
vast.ai
=======================================

ssh-keygen -t rsa
ssh-add; ssh-add -l
cat ~/.ssh/id_rsa.pub

ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCBQ1qHPekHnWWveLwOY25OTBtbrZFrWmHC+7h7lDtqrS/umDNHweybsJLO1Lr3LrbX1212GiqgCQvbBjiibgl2paQjJvHcgdhURdqeUKrnBZq2FUgBRc3z/SeFR1UY4woNNADGKwSNBUcJ7YD3hBfxLbDsRe6bHoFKZoqGtgNMl5YP47EGokM6GqoYgEyqmC27RbIxBmoFfgthFRuBJ4svbOlK/DGD1umECRMuwMziUS5Iruc+Mz9cUU2TJIAqkKFMUkivwI9Bw5BG9p3L4ETr3u4b1+M8IdS2/MEsoyRvau0EI3x9xVF4YRLRyf31M+W/Z8MlQB1/mq2XxiqC4abRCdruEcdFa54eChO1ENBzsOqmt4yIM4IX6L2UcjSlAvciANo7n97hJqxCMDzlhCfYa3duzMZ7ElwAaJTXLQuC48AhAhTUyvAVMvISQQTpytR0NdFwXsuvCEVDbMGjCmhxazguRt6a7QO8KW6qnmmnzt7fiDXrGAUxLzYsDI+OTT8= david@camden


ssh -p 50440 root@145.14.10.31 -L 8080:localhost:8080

export LD_LIBRARY_PATH=/usr/local/cuda-12.0/targets/x86_64-linux/lib
























###################################################################################################
###################################################################################################
###################################################################################################

#### RSYNC SEND ####
rsync -azP -e "ssh -i $LAMBDA_PEM" /home/david/code/davidsvaughn/LLM-utils/llama2/code/*.* $LIP:/home/ubuntu/code
rsync -azP -e "ssh -i $LAMBDA_PEM" /home/david/code/davidsvaughn/LLM-utils/llama2/code/data $LIP:/home/ubuntu/code

# SETUP
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
pip3 install -r requirements.txt

##### FLASH ATTENTION #####
# [ https://www.philschmid.de/instruction-tune-llama-2 ]
## python -c "import torch; assert torch.cuda.get_device_capability()[0] >= 8, 'Hardware not supported for Flash Attention'"
## MAX_JOBS=4 pip install flash-attn --no-build-isolation
pip3 install flash-attn --no-build-isolation

###rm -r data/train
sudo find / -iname "libcudart.so"

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

export HUG_READ_TOKEN=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
huggingface-cli login --token $HUG_READ_TOKEN
export HUG_WRITE_TOKEN=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
huggingface-cli login --token $HUG_WRITE_TOKEN

# RUN
python llama2_train.py

python llama2_train_flash.py

python -i llama2_train_flash.py

python
exec(open('llama2_train_flash.py').read())


#### RSYNC RETRIEVE ####

rsync -azP -e "ssh -i $LAMBDA_PEM" $LIP:/home/ubuntu/code/output-70b/*.* /media/david/toshiba2T/LLM/llama2/results/sep-1-2023/output-70b
rsync -azP -e "ssh -i $LAMBDA_PEM" $LIP:/home/ubuntu/code/output-70b/checkpoint-300 /media/david/toshiba2T/LLM/llama2/results/sep-1-2023/output-70b
rsync -azP -e "ssh -i $LAMBDA_PEM" $LIP:/home/ubuntu/code/output-70b/*.txt /media/david/toshiba2T/LLM/llama2/results/sep-1-2023/output-70b

rsync -azP -e "ssh -i $LAMBDA_PEM" $LIP:/home/ubuntu/code/output-13b/*.* /media/david/toshiba2T/LLM/llama2/results/sep-1-2023/output-13b
rsync -azP -e "ssh -i $LAMBDA_PEM" $LIP:/home/ubuntu/code/output-13b/checkpoint-300 /media/david/toshiba2T/LLM/llama2/results/sep-1-2023/output-13b

rsync -azP -e "ssh -i $LAMBDA_PEM" $LIP:/home/ubuntu/code/output-13b/*.txt /media/david/toshiba2T/LLM/llama2/results/sep-1-2023/output-13b


--------------------------

SEND CHECKPOINT:

rsync -azP -e "ssh -i $LAMBDA_PEM" /media/david/toshiba2T/LLM/llama2/results/aug-29-2023/output-70b/checkpoint-400 $LIP:/home/ubuntu/code/output-70b

rsync -azP -e "ssh -i $LAMBDA_PEM" /media/david/toshiba2T/LLM/llama2/results/sep-1-2023/output-13b/checkpoint-300 $LIP:/home/ubuntu/code/output-13b

rsync -azP -e "ssh -i $LAMBDA_PEM" /media/david/toshiba2T/LLM/llama2/results/aug-29-2023/output-7b/checkpoint-400 $LIP:/home/ubuntu/code/output-7b


SEND MISSING_FIDS:
rsync -azP -e "ssh -i $LAMBDA_PEM" /media/david/toshiba2T/LLM/llama2/results/missing_fids/missing_fids_1.txt $LIP:/home/ubuntu/code

GET RESULTS:

rsync -azP -e "ssh -i $LAMBDA_PEM" $LIP:/home/ubuntu/code/output-7b/*.txt /media/david/toshiba2T/LLM/llama2/results/missing_fids
rsync -azP -e "ssh -i $LAMBDA_PEM" $LIP:/home/ubuntu/code/output-13b/*.txt /media/david/toshiba2T/LLM/llama2/results/missing_fids

==============================================
ssh -i $LAMBDA_PEM ubuntu@209.20.157.12
	LOAD_MODEL_CHKPT = 300
	args.split = -0.5
	shuffle data

----------------------------------------------
ssh -i $LAMBDA_PEM ubuntu@209.20.159.241
	LOAD_MODEL_CHKPT = 400
	args.split = 0.5
	NOT shuffle data

	FIX:
		correct:
			161/9136 : 55415_5.7
			171/9136 : 108802_1.5

		wrong:
			171/9687 : 61793_12.5
			181/9687 : 62033_14.8
			191/9687 : 107111_13.1
==============================================

df -h
pip install huggingface_hub[cli]
huggingface-cli scan-cache
huggingface-cli delete-cache


==============================================================
== LAMBDA : A10==

LLAMAv2_13b

chmod 400 $LAMBDA_PEM
ssh -i $LAMBDA_PEM ubuntu@209.20.157.100
----------------------------------

virtualenv -p python3.8 venv
source venv/bin/activate
mkdir code && cd code

## COPY DATA and CODE (local)
rsync -azP -e "ssh -i $LAMBDA_PEM" /home/david/code/davidsvaughn/LLM-utils/llama2/code/*.* ubuntu@146.235.225.82:/home/ubuntu/code
rsync -azP -e "ssh -i $LAMBDA_PEM" /home/david/code/davidsvaughn/LLM-utils/llama2/code/data ubuntu@146.235.225.82:/home/ubuntu/code

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
pip3 install -r requirements.txt

###rm -r data/train
sudo find / -iname "libcudart.so"

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export HUG_TOKEN=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
huggingface-cli login --token $HUG_TOKEN

python llama2_train.py





==============================================================

# https://www.philschmid.de/sagemaker-llama2-qlora

virtualenv -p python3.10 venv-llama2
source venv-llama2/bin/activate

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

pip install "transformers==4.31.0" "datasets[s3]==2.13.0" sagemaker --upgrade --quiet
pip install spyder matplotlib

export HUG_TOKEN=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
huggingface-cli login --token $HUG_TOKEN

export AWS_PROFILE=dsv_sage_exec
































=====================================================================================================================
=====================================================================================================================
FLAN-T5
=====================================================================================================================
=====================================================================================================================

==============================================================
== AWS ==

		Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04) 20230530
		ami-0705983c654abda59 (64-bit (x86))

chmod 400 g52x.pem

ssh -i g52x.pem ubuntu@3.80.223.9

source activate pytorch

# install Hugging Face Libraries
pip install "peft==0.2.0"
pip install tensorboard matplotlib
pip install "transformers==4.27.2" "datasets==2.9.0" "accelerate==0.17.1" "evaluate==0.4.0" "bitsandbytes==0.37.1" loralib --upgrade

# install additional dependencies needed for training
## pip install rouge-score py7zr

---------------------------------------------------------------

## COPY DATA and CODE (local)
rsync -azP -e "ssh -i g52x.pem" /home/david/code/davidsvaughn/LLM-utils/flant5/code ubuntu@3.80.223.9:/home/ubuntu

## COPY CODE ONLY (local)
rsync -azP -e "ssh -i g52x.pem" /home/david/code/davidsvaughn/LLM-utils/flant5/code/*.py ubuntu@3.80.223.9:/home/ubuntu/code

cd code

python flant5_train.py

python flant5_infer.py

## RETRIEVE MODEL

rsync -azP -e "ssh -i g52x.pem" ubuntu@3.80.223.9:/home/ubuntu/code /home/david/code/davidsvaughn/LLM-utils/flant5/results/g52x_2








==============================================================
== LAMBDA : H100==


chmod 400 $LAMBDA_PEM
ssh -i $LAMBDA_PEM ubuntu@209.20.158.180
----------------------------------

virtualenv -p python3.8 venv
source venv/bin/activate

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
# pip3 install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu118


# install Hugging Face Libraries
pip install "peft==0.2.0"
pip install tensorboard matplotlib
pip install "transformers==4.27.2" "datasets==2.9.0" "accelerate==0.17.1" "evaluate==0.4.0" "bitsandbytes==0.37.1" loralib --upgrade

mkdir code

## COPY DATA and CODE (local)
rsync -azP -e "ssh -i $LAMBDA_PEM" /home/david/code/davidsvaughn/LLM-utils/flant5/code/*.py ubuntu@209.20.158.180:/home/ubuntu/code
rsync -azP -e "ssh -i $LAMBDA_PEM" /home/david/code/davidsvaughn/LLM-utils/flant5/code/data ubuntu@209.20.158.180:/home/ubuntu/code

cd code
rm -r data/train

sudo find / -iname "libcudart.so"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
echo $LD_LIBRARY_PATH

python flant5_train.py




==============================================================
== LAMBDA : A10==


chmod 400 $LAMBDA_PEM
ssh -i $LAMBDA_PEM ubuntu@192.9.226.234
----------------------------------

virtualenv -p python3.8 venv
source venv/bin/activate

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
# pip3 install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu118


# install Hugging Face Libraries
pip install "peft==0.2.0"
pip install tensorboard matplotlib
pip install "transformers==4.27.2" "datasets==2.9.0" "accelerate==0.17.1" "evaluate==0.4.0" "bitsandbytes==0.37.1" loralib --upgrade

mkdir code

## COPY DATA and CODE (local)
rsync -azP -e "ssh -i $LAMBDA_PEM" /home/david/code/davidsvaughn/LLM-utils/flant5/code/*.py ubuntu@192.9.226.234:/home/ubuntu/code
rsync -azP -e "ssh -i $LAMBDA_PEM" /home/david/code/davidsvaughn/LLM-utils/flant5/code/data ubuntu@192.9.226.234:/home/ubuntu/code

cd code
rm -r data/train

sudo find / -iname "libcudart.so"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
echo $LD_LIBRARY_PATH

python flant5_train.py














































































=====================================================================================================================
=====================================================================================================================
=====================================================================================================================
=====================================================================================================================
=====================================================================================================================
=====================================================================================================================

[ https://github.com/tloen/alpaca-lora/issues/174 ]

# jupyter setup
wget http://repo.continuum.io/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh
source ~/.bashrc

conda create --name cap
conda activate cap
conda install pip
conda install cudatoolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

sudo find / -iname "libcudart.so"
  /home/ubuntu/anaconda3/pkgs/cudatoolkit-11.8.0-h6a678d5_0/lib/libcudart.so
  /home/ubuntu/anaconda3/envs/cap/lib/libcudart.so
  /usr/lib/x86_64-linux-gnu/libcudart.so 

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/home/ubuntu/anaconda3/envs/cap/lib:$LD_LIBRARY_PATH

sudo rm -r bitsandbytes
git clone https://github.com/timdettmers/bitsandbytes.git
cd bitsandbytes
CUDA_VERSION=118 make cuda11x   [CUDA_VERSION=116 make cuda116]
python setup.py install

pip install scipy
python -m bitsandbytes
# should be successfull build



pip install "peft==0.2.0"
pip install tensorboard matplotlib
pip install "transformers==4.27.2" "datasets==2.9.0" "accelerate==0.17.1" "evaluate==0.4.0" loralib --upgrade

pip install transformers accelerate --upgrade

cd code
python flant5_train.py


---------------------------------------------------------------
***** WORKED ******
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
chmod u+x Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
source ~/.bashrc

conda create -n conda39 python=3.9
conda activate conda39
conda install pip

[ /home/ubuntu/miniconda3/envs/conda39/bin/pip ]
pip install torch==2.0.0+cu118 torchaudio==2.0.0+cu118 torchvision==0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install the other CUDA packages
conda install cudnn
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install -c "nvidia/label/cuda-11.8.0" cuda-runtime

export LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/conda39/lib

sudo rm -r bitsandbytes
sudo git clone https://github.com/timdettmers/bitsandbytes.git
sudo chmod 777 -R bitsandbytes
cd bitsandbytes
CUDA_VERSION=118 make cuda11x   [CUDA_VERSION=116 make cuda116]
python setup.py install

pip install "peft==0.2.0"
pip install tensorboard matplotlib
pip install "transformers==4.27.2" "datasets==2.9.0" "accelerate==0.17.1" "evaluate==0.4.0" loralib --upgrade
pip install scipy

python -m bitsandbytes

pip install --upgrade --force-reinstall pyarrow==11.0.0





























































-------------------------------------------------------
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
sudo sh cuda_11.7.1_515.65.01_linux.

			Driver:   Not Selected
			Toolkit:  Installed in /usr/local/cuda-11.7/

			Please make sure that
			 -   PATH includes /usr/local/cuda-11.7/bin
			 -   LD_LIBRARY_PATH includes /usr/local/cuda-11.7/lib64, or, add /usr/local/cuda-11.7/lib64 to /etc/ld.so.conf and run ldconfig as root

			To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.7/bin

virtualenv venv
source venv/bin/activate
pip3 install torch torchvision torchaudio

export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64
















-------------------
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run

			Please make sure that
			 -   PATH includes /usr/local/cuda-12.2/bin
			 -   LD_LIBRARY_PATH includes /usr/local/cuda-12.2/lib64, or, add /usr/local/cuda-12.2/lib64 to /etc/ld.so.conf and run ldconfig as root

------------------------------------------------------

# install anaconda
sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash /tmp/Anaconda3-2020.02-Linux-x86_64.sh
source ~/.bashrc

# create conda env
[ conda update -n base -c defaults conda ]
conda create -n venv python=3.8 anaconda
conda activate venv
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# fix pip
which -a pip
conda install pip
which -a pip

# COPY DATA (local)
rsync -azP -e "ssh -i $LAMBDA_PEM" /home/david/code/davidsvaughn/LLM-utils/flant5/code ubuntu@150.230.39.77:/home/ubuntu

# RUN
cd code
python flant5_feedback.py



---------------------------------------------------------------
# https://docs.lambdalabs.com/linux/install-docker-run-container/

sudo apt -y update && sudo apt -y install docker.io nvidia-container-toolkit && \
sudo systemctl daemon-reload && \
sudo systemctl restart docker

			The following packages have unmet dependencies:
			 docker.io : Depends: containerd (>= 1.2.6-0ubuntu1~)
			E: Unable to correct problems, you have held broken packages.

sudo adduser "$(id -un)" docker

docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
docker images
#  docker run --gpus all -it --rm 71eb2d092138
docker run -d --gpus all --entrypoint "top" 71eb2d092138 -b
docker cp code/. dazzling_torvalds:/workspace/code
docker exec -it dazzling_torvalds /bin/bash

---------------------------------------------------------------
# install Hugging Face Libraries
pip install "peft==0.2.0"
pip install tensorboard matplotlib
pip install "transformers==4.27.2" "datasets==2.9.0" "accelerate==0.17.1" "evaluate==0.4.0" "bitsandbytes==0.37.1" loralib --upgrade

---------------------------------------------------------------

## COPY DATA (local)
rsync -azP -e "ssh -i $LAMBDA_PEM" /home/david/code/davidsvaughn/LLM-utils/flant5/code ubuntu@209.20.158.78:/home/ubuntu

cd code
python flant5_feedback.py























=====================================================================================================================
== ERRORS ==

CUDA SETUP: CUDA runtime path found: /opt/conda/envs/pytorch/lib/libcudart.so
CUDA SETUP: Highest compute capability among GPUs detected: 8.6
CUDA SETUP: Detected CUDA version 117
CUDA SETUP: Loading binary /opt/conda/envs/pytorch/lib/python3.9/site-packages/bitsandbytes/libbitsandbytes_cuda117.so...
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:16<00:00,  1.36s/it]
trainable params: 18874368 || all params: 11154206720 || trainable%: 0.16921300163961817
/opt/conda/envs/pytorch/lib/python3.9/site-packages/transformers/optimization.py:391: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning


========================================================================

virtualenv venv
source venv/bin/activate

python -m pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0+cu115 -f https://download.pytorch.org/whl/cu115/torch_stable.html


python -m pip install torch==1.11.0+cu115 -f https://download.pytorch.org/whl/cu115/torch_stable.html


-------------------------

ubuntu@146-235-237-171:~/code$ echo $PATH
/home/ubuntu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

ubuntu@146-235-237-171:~/code$ sudo find / -name "libcudart.so"
/usr/lib/x86_64-linux-gnu/libcudart.so

ubuntu@146-235-237-171:~/code$ sudo find / -name "lib64"
/usr/src/linux-headers-5.4.0-139/arch/sh/lib64
/usr/lib64
/usr/lib/cuda/lib64
/usr/lib/nvidia-cuda-toolkit/lib64
/snap/core20/1822/lib64
/snap/core20/1822/usr/lib64
/snap/core20/1950/lib64
/snap/core20/1950/usr/lib64
/snap/snapd/19457/lib64
/snap/snapd/18357/lib64
/lib64



export LD_LIBRARY_PATH=/usr/lib/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/usr/lib/x86_64-linux-gnu${PATH:+:${PATH}}
















===========================================================================================================


===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
CUDA_SETUP: WARNING! libcudart.so not found in any environmental path. Searching /usr/local/cuda/lib64...
/home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/cuda_setup/main.py:136: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/usr/local/cuda/lib64')}
  warn(msg)
ERROR: python: undefined symbol: cudaRuntimeGetVersion
CUDA SETUP: libcudart.so path is None
CUDA SETUP: Is seems that your cuda installation is not in your path. See https://github.com/TimDettmers/bitsandbytes/issues/85 for more information.
CUDA SETUP: CUDA version lower than 11 are currently not supported for LLM.int8(). You will be only to use 8-bit optimizers and quantization routines!!
/home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/cuda_setup/main.py:136: UserWarning: WARNING: No libcudart.so found! Install CUDA or the cudatoolkit package (anaconda)!
  warn(msg)
CUDA SETUP: Highest compute capability among GPUs detected: 8.6
CUDA SETUP: Detected CUDA version 00
CUDA SETUP: Loading binary /home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so...
/home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/cextension.py:31: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
Loading checkpoint shards:   0%|                                                                                                                           | 0/12 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "flant5_feedback.py", line 231, in <module>
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
  File "/home/ubuntu/.local/lib/python3.8/site-packages/transformers/models/auto/auto_factory.py", line 471, in from_pretrained
    return model_class.from_pretrained(
  File "/home/ubuntu/.local/lib/python3.8/site-packages/transformers/modeling_utils.py", line 2647, in from_pretrained
    ) = cls._load_pretrained_model(
  File "/home/ubuntu/.local/lib/python3.8/site-packages/transformers/modeling_utils.py", line 2970, in _load_pretrained_model
    new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
  File "/home/ubuntu/.local/lib/python3.8/site-packages/transformers/modeling_utils.py", line 676, in _load_state_dict_into_meta_model
    set_module_8bit_tensor_to_device(model, param_name, param_device, value=param)
  File "/home/ubuntu/.local/lib/python3.8/site-packages/transformers/utils/bitsandbytes.py", line 70, in set_module_8bit_tensor_to_device
    new_value = bnb.nn.Int8Params(new_value, requires_grad=False, has_fp16_weights=has_fp16_weights).to(device)
  File "/home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/nn/modules.py", line 196, in to
    return self.cuda(device)
  File "/home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/nn/modules.py", line 160, in cuda
    CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
  File "/home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/functional.py", line 1616, in double_quant
    row_stats, col_stats, nnz_row_ptr = get_colrow_absmax(
  File "/home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/functional.py", line 1505, in get_colrow_absmax
    lib.cget_col_row_stats(ptrA, ptrRowStats, ptrColStats, ptrNnzrows, ct.c_float(threshold), rows, cols)
  File "/usr/lib/python3.8/ctypes/__init__.py", line 386, in __getattr__
    func = self.__getitem__(name)
  File "/usr/lib/python3.8/ctypes/__init__.py", line 391, in __getitem__
    func = self._FuncPtr((name_or_ordinal, self))
AttributeError: /home/ubuntu/.local/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so: undefined symbol: cget_col_row_stats
