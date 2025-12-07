# Ultralytics
## Setup Environment
This repository use anaconda to specify version of python for preventing error from deprecated package conflicts.
### Download miniconda package
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
### install miniconda package
```bash
bash ~/Miniconda3-latest-Linux-x86_64.sh
```

### verify installation
```bash
source ~/.bashrc
conda init --all
conda --version
```

### Create conda environment with python=3.13
```bash
conda create --name ult python=3.13
conda activate ult
```

### install ultralytics
```bash
pip install ultralytics
```

## Example Folder Structure
```bash
├── RT-Detr/
    ├── datasets
    ├── checkpoints
    ├── runs
    ├── train.py
    ├── inference.py
```

## Build
```bash
docker buil -t ult .
docker tun ult
```

## Run
### Train
Example trained already provided in train.ps1
```bash
train.ps1
```
