# MatBERT
A pretrained BERT model on materials science literature.

## Downloading data files

MatBERT is now "semi-public" in the sense that everyone can download 
the model files, but we don't want too many people have access these
download links now.

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

The model can be loaded using Transformers' unversal loading API:

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

## Training details

Training of all MatBERT models was done using `transformers==3.3.1`.
The corpus of this training contains 2 million papers collected by the
text-mining efforts at [CEDER group](https://ceder.berkeley.edu/). In
total, we had collected 61,253,938 paragraphs, from which around 50 million
paragraphs with 20-510 tokens are filtered and used for training. Two WordPiece
tokenizers (cased and uncased) that are optimized for materials science 
literature was trained using these paragraphs.

For training MatBERT, the config files we used were [bert-base-uncased](matbert/training/configs/bert-base-uncased-wd.json)
and [bert-base-cased](matbert/training/configs/bert-base-cased-wd.json).
Only the masked language modeling (MLM) task was used to pretrain MatBERT models.
Roughly the batch size is 192 paragraphs per gradient update step and there are
5 epochs in total. The optimizer used is Adam with beta1=0.9 and beta2=0.999. 
Learning rates start with 5e-5 and decays linearly to zero as the training finishes. 
A weight decay of 0.01 was used. All models are trained using FP16 mode and O2 
optimization on 8 NVIDIA V100 cards.

## Citing

Coming soon, stay tuned.
