# MatBERT
A pretrained BERT model on materials science literature

## Downloading data files

MatBERT is now "semi-public" in the sense that everyone can download 
the model files, but we don't want too many people have access these
download links now.

To use MatBERT, first download these files into a folder:

```
export MODEL_PATH="Your path"
curl -# -o $MODEL_PATH/config.json https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/config.json
curl -# -o $MODEL_PATH/vocab.txt https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/vocab.txt
curl -# -o $MODEL_PATH/pytorch_model.bin https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/pytorch_model.bin
```

Then, load the model using Transformers:

```python
>>> from transformers import pipeline
>>> from pprint import pprint

>>> unmasker = pipeline('fill-mask', model='Your path')
>>> pprint(unmasker("Conventional [MASK] synthesis is used to fabricate material LiMn2O4."))
[{'score': 0.2966659665107727,
  'sequence': '[CLS] conventional hydrothermal synthesis is used to fabricate '
              'material limn2o4. [SEP]',
  'token': 7524,
  'token_str': 'hydrothermal'},
 {'score': 0.15518330037593842,
  'sequence': '[CLS] conventional chemical synthesis is used to fabricate '
              'material limn2o4. [SEP]',
  'token': 2868,
  'token_str': 'chemical'},
 {'score': 0.09272155165672302,
  'sequence': '[CLS] conventional solvothermal synthesis is used to fabricate '
              'material limn2o4. [SEP]',
  'token': 17831,
  'token_str': 'solvothermal'},
 {'score': 0.08843386173248291,
  'sequence': '[CLS] conventional combustion synthesis is used to fabricate '
              'material limn2o4. [SEP]',
  'token': 5444,
  'token_str': 'combustion'},
 {'score': 0.04024626314640045,
  'sequence': '[CLS] conventional thermal synthesis is used to fabricate '
              'material limn2o4. [SEP]',
  'token': 2880,
  'token_str': 'thermal'}]
```

## Details of the training

Training of all MatBERT models was done using `transformers==3.3.1`.
The corpus of this training contains 2 million papers collected by the
text-mining efforts at [CEDER group](https://ceder.berkeley.edu/). In
total, we had collected 61,253,938 paragraphs, from which around 50 million
paragraphs with 20-510 tokens are filtered and used for training. Two WordPiece
tokenizers (cased and uncased) that are optimized for materials science 
literature was trained using these paragraphs.

For training MatBERT, the config files we used were [bert-base-uncased](matbert/training/configs/bert-base-uncased.json)
and [bert-base-cased](matbert/training/configs/bert-base-cased.json).
Only the masked language modeling (MLM) task was used to pretrain MatBERT models.
Roughly the batch size is 192 paragraphs per gradient update step and there are
5 epochs in total. Learning rates start with 5e-5 and decays linearly to zero
as the training finishes. All models are trained using FP16 mode and O2 optimization 
on 8 NVIDIA V100 cards.

