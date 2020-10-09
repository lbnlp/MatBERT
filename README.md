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