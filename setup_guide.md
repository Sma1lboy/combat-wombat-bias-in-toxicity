# Combat Wombat Bias Detector - Setup Guide

This guide will help you set up the environment to run the bias detection models.

## Requirements

The project requires the following Python packages:

- numpy (>=1.16.0)
- pandas (>=0.24.0)
- PyTorch (>=1.0.0)
- NVIDIA Apex (0.1)
- tensorboardX (>=1.6)
- pytorch_pretrained_bert (0.6.2)
- scikit-learn (>=0.20.0)
- gensim (>=3.7.0)

## Environment Setup

1. Create a virtual environment (recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Install NVIDIA Apex for mixed precision training:

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Data Setup

1. Download the dataset from the Jigsaw Unintended Bias in Toxicity Classification competition and place it in the following directory structure:

```
../input/jigsaw-unintended-bias-in-toxicity-classification/
    - train.csv
    - test.csv
```

## Directory Structure

Ensure you have the following directory structure:

```
./
├── code/
│   ├── toxic/  # Local modules
│   └── train_bert_2_uncased.py
├── models/     # Where trained models will be saved
└── tb_logs/    # TensorBoard logs
```

## Running the Models

To train the BERT models with the provided configurations:

```
python code/train_bert_2_uncased.py
```

The script will train three different BERT model configurations. Models will be saved to the `./models/` directory.

## Hardware Requirements

- CUDA-compatible GPU with at least 16GB memory
- At least 32GB RAM for multiprocessing
- SSD storage recommended for faster data loading
