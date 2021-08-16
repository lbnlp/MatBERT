# MatBERT
A pretrained BERT model on materials science literature. MatBERT specializes in understanding 
materials science terminologies and paragraph-level scientific reasoning.

## Downloading data files

To use MatBERT, download these files into a folder:

```
export MODEL_PATH="Your path"
mkdir $MODEL_PATH/matbert-base-cased $MODEL_PATH/matbert-base-uncased

curl -# -o $MODEL_PATH/matbert-base-cased/config.json https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/config.json
curl -# -o $MODEL_PATH/matbert-base-cased/vocab.txt https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/vocab.txt
curl -# -o $MODEL_PATH/matbert-base-cased/pytorch_model.bin https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/pytorch_model.bin

curl -# -o $MODEL_PATH/matbert-base-uncased/config.json https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_uncased_30522_wd/config.json
curl -# -o $MODEL_PATH/matbert-base-uncased/vocab.txt https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_uncased_30522_wd/vocab.txt
curl -# -o $MODEL_PATH/matbert-base-uncased/pytorch_model.bin https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_uncased_30522_wd/pytorch_model.bin
```

## Using MatBERT

### Tokenizers

The tokenizer is specifically trained to handle materials science terminologies:

```python
>>> from transformers import BertTokenizerFast
>>> tokenizer = BertTokenizerFast.from_pretrained('PATH-TO-MATBERT/matbert-base-cased', do_lower_case=False)
>>> tokenizer_bert = BertTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False)

>>> for i in ['Fe(NO3)3• 9H2O', 'La0.85Ag0.15Mn1−yAlyO3']:
>>>     print(i)
>>>     print('='*100)
>>>     print('MatBERT tokenizer:', tokenizer.tokenize(i))
>>>     print('BERT tokenizer:', tokenizer_bert.tokenize(i))

Fe(NO3)3• 9H2O
====================================================================================================
MatBERT tokenizer: ['Fe', '(', 'NO3', ')', '3', '•', '9H2O']
BERT tokenizer: ['Fe', '(', 'NO', '##3', ')', '3', '•', '9', '##H', '##2', '##O']
La0.85Ag0.15Mn1−yAlyO3
====================================================================================================
MatBERT tokenizer: ['La0', '.', '85', '##Ag', '##0', '.', '15', '##Mn1', '##−y', '##Al', '##y', '##O3']
BERT tokenizer: ['La', '##0', '.', '85', '##A', '##g', '##0', '.', '15', '##M', '##n', '##1', '##−', '##y', '##A', '##ly', '##O', '##3']
```

### The model

The model can be loaded using Transformers' unversal loading API. Here, we demonstrate
how MatBERT performs scientific reasoning for the synthesis of Li-ion battery materials.

```python
>>> from transformers import BertForMaskedLM, BertTokenizerFast, pipeline
>>> from pprint import pprint

>>> model = BertForMaskedLM.from_pretrained('PATH-TO-MATBERT/matbert-base-cased')
>>> tokenizer = BertTokenizerFast.from_pretrained('PATH-TO-MATBERT/matbert-base-cased', do_lower_case=False)
>>> unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
>>> pprint(unmasker("Conventional [MASK] synthesis is used to fabricate material LiMn2O4."))
[{'sequence': '[CLS] Conventional combustion synthesis is used to fabricate material LiMn2O4. [SEP]',
  'score': 0.4971400499343872,
  'token': 5444,
  'token_str': 'combustion'},
 {'sequence': '[CLS] Conventional hydrothermal synthesis is used to fabricate material LiMn2O4. [SEP]',
  'score': 0.2478722780942917,
  'token': 7524,
  'token_str': 'hydrothermal'},
 {'sequence': '[CLS] Conventional chemical synthesis is used to fabricate material LiMn2O4. [SEP]',
  'score': 0.060953784734010696,
  'token': 2868,
  'token_str': 'chemical'},
 {'sequence': '[CLS] Conventional gel synthesis is used to fabricate material LiMn2O4. [SEP]',
  'score': 0.03871171176433563,
  'token': 4003,
  'token_str': 'gel'},
 {'sequence': '[CLS] Conventional solution synthesis is used to fabricate material LiMn2O4. [SEP]',
  'score': 0.019403140991926193,
  'token': 2291,
  'token_str': 'solution'}]
```

## Evaluation

### GLUE

The [General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/) 
is a collection of resources for training, evaluating, and analyzing natural language understanding systems.
Note, GLUE evaluates language models' capability to model general purpose language understanding,
which may not align with the capabilities of language models trained on special domains, such as MatBERT.

| Task                                   | Metric                | Score (MatBERT-base-cased) | Score (MatBERT-base-uncased) |
|----------------------------------------|-----------------------|---------------------------:|-----------------------------:|
| The Corpus of Linguistic Acceptability | Matthew's Corr        |                       26.1 |                         26.6 |
| The Stanford Sentiment Treebank        | Accuracy              |                       89.5 |                         90.2 |
| Microsoft Research Paraphrase Corpus   | F1/Accuracy           |                  87.0/83.1 |                    87.4/82.7 |
| Semantic Textual Similarity Benchmark  | Pearson-Spearman Corr |                  80.3/79.3 |                    81.5/80.2 |
| Quora Question Pairs                   | F1/Accuracy           |                  69.8/88.4 |                    69.7/88.6 |
| MultiNLI Matched                       | Accuracy              |                       79.6 |                         80.7 |
| MultiNLI Mismatched                    | Accuracy              |                       79.3 |                         80.1 |
| Question NLI                           | Accuracy              |                       88.4 |                         88.5 |
| Recognizing Textual Entailment         | Accuracy              |                       63.0 |                         60.2 |
| Winograd NLI                           | Accuracy              |                       61.6 |                         65.1 |
| Diagnostics Main                       | Matthew's Corr        |                       32.6 |                         31.9 |
| Average Score                          | ----                  |                       72.4 |                         72.9 |


## Training details

Training of all MatBERT models was done using `transformers==3.3.1`.
The corpus of this training contains 2 million papers collected by the
text-mining efforts at [CEDER group](https://ceder.berkeley.edu/). In
total, we had collected 61,253,938 paragraphs, from which around 50 million
paragraphs with 20-510 tokens are filtered and used for training. Two WordPiece
tokenizers (cased and uncased) that are optimized for materials science 
literature was trained using these paragraphs.

The DOIs and titles of the 2 million papers can be found at [this CSV file](https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/PretrainingDataDOIs.csv.zip).
To grasp an overview of this corpus, we created a word cloud image [here](docs/PretrainingCorpusWordCloud.png).

For training MatBERT, the config files we used were 
[matbert-base-uncased](matbert/training/configs/bert-base-uncased-wd.json)
and [matbert-base-cased](matbert/training/configs/bert-base-cased-wd.json).
Only the masked language modeling (MLM) task was used to pretrain MatBERT models.
Roughly the batch size is 192 paragraphs per gradient update step and there are
5 epochs in total. The optimizer used is AdamW with beta1=0.9 and beta2=0.999. 
Learning rates start with 5e-5 and decays linearly to zero as the training finishes. 
A weight decay of 0.01 was used. All models are trained using FP16 mode and O2 
optimization on 8 NVIDIA V100 cards. The loss values during training can be found
at [matbert-base-uncased](docs/model_2Mpapers_uncased_30522_wd.csv) and
[matbert-base-cased](docs/model_2Mpapers_cased_30522_wd.csv).

## Citing

Coming soon, stay tuned.

This work used the Extreme Science and Engineering Discovery Environment (XSEDE) 
GPU resources, specifically the Bridges-2 supercomputer at the Pittsburgh 
Supercomputing Center, through allocation TG-DMR970008S.
