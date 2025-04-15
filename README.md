# COMBAT WOMBAT [4th place solution to the Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/leaderboard)

You should be able to replicate the solution and retrain all the models from [our inference kernel](https://www.kaggle.com/iezepov/wombat-inference-kernel) just by running all `train_*.py` scripts. One would need to put the input data and [the embeddings dataset](https://www.kaggle.com/iezepov/gensim-embeddings-dataset) to the `input` folder.

`code/toxic` contains various utils that are used in `train_*` files.

## Data Setup

Before running the code, you need to download the following data:

1. **Competition Dataset**: Download the training and test data from [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) competition on Kaggle.

2. **Embeddings Dataset**: Download the [Gensim Embeddings Dataset](https://www.kaggle.com/iezepov/gensim-embeddings-dataset) from Kaggle.

Place the downloaded files in the following directory structure:

```
input/
  ├── jigsaw-unintended-bias-in-toxicity-classification/
  │     ├── train.csv
  │     └── test.csv
  │
  └── gensim-embeddings-dataset/
        ├── glove.840B.300d.gensim
        ├── crawl-300d-2M.gensim
        ├── paragram_300_sl999.gensim
        └── GoogleNews-vectors-negative300.gensim
```

If you're running with limited RAM, you can modify `code/train_lstms.py` to use randomly initialized embeddings instead of loading the pretrained ones.

## System Requirements

### Hardware Requirements

- **GPU version**: CUDA-compatible GPU with at least 16GB VRAM
- **CPU-only version**: At least 32GB RAM
- SSD storage recommended for faster data loading

### PyTorch Requirements

Two installation options are available:

- **GPU version**: `torch==1.4.0+cu101` (CUDA 10.1 support)
- **CPU version**: `torch==1.4.0+cpu`

### Complete Environment

For GPU environments:

```
torch==1.4.0+cu101
torchvision==0.5.0
fastai==1.0.60
```

For CPU-only environments:

```
torch==1.4.0+cpu
torchvision==0.5.0
fastai==1.0.60
```

Install specific PyTorch versions:

```
# GPU version
pip install torch==1.4.0+cu101 torchvision==0.5.0

# CPU version
pip install torch-1.4.0+cpu-cp37-cp37m-win_amd64.whl torchvision==0.5.0
```

All other requirements are listed in `requirements.txt`.

## Environment Setup

### Python Environment

- Python 3.7 (recommended and tested)

### Installation Instructions

#### Using requirements.txt (Recommended)

We provide a `requirements.txt` file with all the necessary dependencies locked to compatible versions. This is the easiest way to set up the environment:

```bash
# Create and activate environment
conda create -n toxic_env python=3.7 -y
conda activate toxic_env

# Install PyTorch (choose ONE of the following options)
# For GPU:
pip install https://download.pytorch.org/whl/cu101/torch-1.4.0-cp37-cp37m-win_amd64.whl
# For CPU:
# pip install torch-1.4.0+cpu-cp37-cp37m-win_amd64.whl

# Install torchvision
pip install torchvision==0.5.0

# Install all other dependencies
pip install -r requirements.txt

# If protobuf causes issues, downgrade it:
pip install protobuf==3.20.3
```

Note: The `requirements.txt` file includes all necessary packages except for PyTorch, which should be installed separately as shown above due to different CPU/GPU variants.

#### GPU Environment Setup (Manual Installation)

```bash
# Create and activate environment
conda create -n toxic_env python=3.7 -y
conda activate toxic_env

# Install PyTorch with CUDA 10.1
pip install https://download.pytorch.org/whl/cu101/torch-1.4.0-cp37-cp37m-win_amd64.whl
pip install torchvision==0.5.0

# Install other dependencies
pip install numpy==1.17.0 pandas==0.25.0 scikit-learn==0.21.0 tqdm==4.38.0 keras==2.2.5 gensim==3.8.3 nltk==3.4.5 fastai==1.0.60 spacy==2.3.7 emoji==0.5.4 fasttext-wheel==0.9.2 pytorch-pretrained-bert==0.6.2 tensorflow==1.14.0 protobuf==3.20.3
```

#### CPU-only Environment Setup

```bash
# Create and activate environment
conda create -n toxic_env python=3.7 -y
conda activate toxic_env

# Install PyTorch CPU version
pip install torch-1.4.0+cpu-cp37-cp37m-win_amd64.whl
pip install torchvision==0.5.0

# Install other dependencies
pip install numpy==1.17.0 pandas==0.25.0 scikit-learn==0.21.0 tqdm==4.38.0 keras==2.2.5 gensim==3.8.3 nltk==3.4.5 fastai==1.0.60 spacy==2.3.7 emoji==0.5.4 fasttext-wheel==0.9.2 pytorch-pretrained-bert==0.6.2 tensorflow==1.14.0 protobuf==3.20.3
```

### Important Notes

- You may see numpy dtype FutureWarning messages when running the code. These are deprecation warnings from TensorFlow and can be safely ignored.
- If you encounter dependency conflicts, try installing packages individually or with the `--no-deps` flag.
- For spaCy models, install using: `python -m spacy download en_core_web_sm`

## Outline of our final solution

We ended up using a simple average ensemble of 33 models:

- 12 LSTM-based models
- 17 BERT models (base only models)
- 2 GPT2-based models

Not-standart things:

- We were predicting 18 targets with evey model
- We combined some of the target to get the final score. It was hurting AUC but was improving the target metric we care about.
- LSTMs were using char-level embeddings
- LSTMs were trained on different set of embeddings
- We mixed three different types of BERTs: cased, uncased and fine-tuned ucnased (we used [the standart fine-tuning procedure](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples/lm_finetuning))
- We tried to unbias our models with some PL. The idea was taken from [this paper, "Bias Mitigation" section](http://www.aies-conference.com/wp-content/papers/main/AIES_2018_paper_9.pdf)
- GPT2 models were using a CNN-based classifier head isntead of the linear classifier head

## The loss

We trained all our models on a quite straightforward BCE loss. the only thing is that we had 18 targets.

- the main target, with special weights that highlites data points that mention identities
- the main taregt, again, but without any extra weights
- 5 toxicity types
- 9 main identites
- max value for any out of 9 identities columns
- binary column that indecates whether at least one identity was mentioned

All targets, except the last one, were used as soft targets - flaot values from 0 to 1, not hard binary 0/1 targets.

We used a somewhat common weights for the first loss. The toxicity subtypes were trained without any special weights. And the identites loss (last 12 targets) were trained with 0 weight for the NA identities. Even thought the NAs were treated like zeros we decided not to trust that information during the training. Can't say whether it worked though, didn't have time to properly run the ablation study.

In order to improve diversity of out ensemble we sometimes sligtly increased or decreased weight of the main part of the loss (`config.main_loss_weight`).

The loss function we used:

```
def custom_loss(data, targets):
    bce_loss_1 = nn.BCEWithLogitsLoss(targets[:, 1:2])(data[:, :1], targets[:, :1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 1:7], targets[:, 2:8])
    bce_loss_3 = nn.BCEWithLogitsLoss(targets[:, 19:20])(data[:, 7:18], targets[:, 8:19])
    return config.main_loss_weight * bce_loss_1 + bce_loss_2 + bce_loss_3 / 4
```

The weights we used for the main part of the loss:

```
iden = train[IDENTITY_COLUMNS].fillna(0).values

weights = np.ones(len(train))
weights += (iden >= 0.5).any(1)
weights += (train["target"].values >= 0.5) & (iden < 0.5).any(1)
weights += (train["target"].values < 0.5) & (iden >= 0.5).any(1)
weights /= weights.mean()  # Don't need to downscale the loss
```

And finally the identites targets and weights:

```
subgroup_target = np.hstack([
    (iden >= 0.5).any(axis=1, keepdims=True).astype(np.int),
    iden,
    iden.max(axis=1, keepdims=True),
])
sub_target_weigths = (
    ~train[IDENTITY_COLUMNS].isna().values.any(axis=1, keepdims=True)
).astype(np.int)
```

## LSTM-based models

### Architecture

We ended up using a very simple architecture with 2 bidirectional LSTM layers and a dense layer on top of that. You defenitely saw this architecture being copy-pased from one public kernel to another :) The only modificatrion we had is taking pooling not only from the top (second) LSTM layer, but from the first layer as well. Average pooling from both layers and max pooling from the last layer weere concatinaed before the dense classifiyng layers.

The diversity of oru 12 LSTM models comes from using different embeddings and different initial random states. We used the following embeddings (most of them you know and lvoe):

- glove.840B.300d (global vectors)
- crawl-300d-2M (fasttext)
- paragram_300_sl999 (paragram)
- GoogleNews-vectors-negative300 (word2vec)
- Char represnation. We computed top-100 most frequent chars in the data and then represented every token with counts on those chars. That's a poor man's char level modeling :)

Every LSTM model was taking 4 out of 5 concatinated embeddings. Thus the diversity.

We found the trick where you lookup various word forms in the mebdding dictionary being very helpful. That what we ended up using:

```
PORTER_STEMMER = PorterStemmer()
LANCASTER_STEMMER = LancasterStemmer()
SNOWBALL_STEMMER = SnowballStemmer("english")

def word_forms(word):
    yield word
    yield word.lower()
    yield word.upper()
    yield word.capitalize()
    yield PORTER_STEMMER.stem(word)
    yield LANCASTER_STEMMER.stem(word)
    yield SNOWBALL_STEMMER.stem(word)

def maybe_get_embedding(word, model):
    for form in word_forms(word):
        if form in model:
            return model[form]

    word = word.strip("-'")
    for form in word_forms(word):
        if form in model:
            return model[form]

    return None

def gensim_to_embedding_matrix(word2index, path):
    model = KeyedVectors.load(path, mmap="r")
    embedding_matrix = np.zeros(
        (max(word2index.values()) + 1, model.vector_size),
        dtype=np.float32,
    )
    for word, i in word2index.items():
        maybe_embedding = maybe_get_embedding(word, model)
        if maybe_embedding is not None:
            embedding_matrix[i] = maybe_embedding
    return embedding_matrix
```

### Preprocessing

We did some preprocessing for the LSTM models and found it helpful:

- Replacements for some weird unicode chars like `\ufeff`
- Replacements for the "starred" words like `bit*h`
- Removing all uncide chars of the `Mn` category
- Replacing all Hebrew chars with the same Hebrew letter `א`. This trick grately reduces the number of distinct characters in the vocabulary. We didn't expect our model to learn toxicity in different languages anyway. But we don't want to lose information about that some Hebrew word was here
- The same logic applied to Arabic, Chineese and Japaneese chars
- Replacing emojies with their aliases and a special `EMJ` . token so the model could understand that an emoji was here.

## BERTs

### It's all about speed

We found that we can greatly increase the inference time for BERT models with two tricks. Both of them are possibly because of the `attention_mask` we ahve in berts. What it means for us is that we can zero-pad our sentences as much as we want, it doesn't matter for BERT. Wem managed to reduces inference time from 17 minutes as is to 3.5 minutes. Almost 5x times!

The first optimization is obvious - pad the whole dataset not to the some constant `MAX_LEN` value (we used 400 btw), but pad it to the longest comment in batch. Or rather clip the batch to remove some dead weight of zeros on the right. We did it like that:

```
def clip_to_max_len(batch):
    X, lengths = map(torch.stack, zip(*batch))
    max_len = torch.max(lengths).item()
    return X[:, :max_len]

# sequences is zero-padded numpy array
lengths = np.argmax(sequences == 0, axis=1)
lengths[lengths == 0] = sequences.shape[1]

sequences = torch.from_numpy(sequences)
lengths = torch.from_numpy(lengths)

test_dataset = data.TensorDataset(sequences, lengths)
test_loader = data.DataLoader(
    test_dataset,
    batch_size=128,
    collate_fn=clip_to_max_len,
)
```

The second is much more important optimization is to sort the dataset by length. This allowed us to greatly reduce the amount of the dead weight of zeros. And just a tiny change to the code above:

```
ids = lengths.argsort(kind="stable")
inverse_ids = test.id.values[ids].argsort(kind="stable")

test_loader = data.DataLoader(
    data.Subset(test_dataset, ids),
    batch_size=128,
    collate_fn=clip_to_max_len,
)

# preds = here_you_get_predictions_from_the_model
preds = preds[inverse_ids]
```

### Bias Mitigation with PL

We took and idea of adding extra non-toxic data that mentions identites to the dadset to kill the bias of the model. The flow is straightforward:

- Take an amazing datasdet of (8M senteces from wikipedia)[https://www.kaggle.com/mikeortman/wikipedia-sentences]
- Run predictions of all 18 targets on them with the best single model
- Filter out senteces that don't mention any identites
- Filter out senteces with high toxicity predictions (some movies and music albums ahve "toxic" titles like `Hang 'em high!`)
- Set the targets for toxicity and subtypes of toxicity to 0 since you know that sentecces from wikipedia should not be toxic
- Add those senteces to the training data to kill the bias towards some identites

According to the article mentioned above the trick works very well. We saw string improvemnts on the CV, which didn't translate to as strong improvent on the LB. We still believe that this is a very neat trick, but one needs to be careful with sampling - to math the distribution of lengths at the very elast, or better, other tokens.

# Combat Wombat Bias in Toxicity Classification

This project uses BERT models to detect and combat unintended bias in toxicity classification.

## Requirements

The code requires the following Python packages:

- numpy (>=1.16.0)
- pandas (>=0.24.0)
- PyTorch (>=1.0.0)
- NVIDIA Apex (0.1)
- tensorboardX (>=1.6)
- pytorch_pretrained_bert (0.6.2)

## Setup

1. Install the required packages:

```
pip install -r requirements.txt
```

2. Install NVIDIA Apex for mixed precision training:

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

3. Download the dataset from the Jigsaw Unintended Bias in Toxicity Classification competition and place it in the `../input/jigsaw-unintended-bias-in-toxicity-classification/` directory.

## Running the Model

To train the BERT models:

```
python code/train_bert_2_uncased.py
```

The script will train three different configurations of the model. The trained models will be saved in the `./models/` directory, and TensorBoard logs will be saved in `./tb_logs/`.
