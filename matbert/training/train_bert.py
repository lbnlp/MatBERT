import functools
import logging
import math
import os
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, List, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    BertConfig, BertTokenizerFast, BertForMaskedLM,
    DataCollatorForLanguageModeling, PreTrainedTokenizer, HfArgumentParser)
from transformers import Trainer, TrainingArguments

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s][%(name)s]: %(message)s",
    datefmt="%m/%d %H:%M:%S",
    level=logging.INFO,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model checkpoint."},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Model name if training from scratch"},
    )


@dataclass
class DataTrainingArguments:
    train_data_dir: Optional[str] = field(
        metadata={"help": "The input training data file (a text file)."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    block_size: int = field(
        default=512,
        metadata={"help": "Size of input tensor size"},
    )


@dataclass
class TrainingOptions:
    output_dir: Optional[str] = field(
        metadata={"help": "Model output dir."}
    )
    checkpoint_dir: Optional[str] = field(
        default=None, metadata={"help": "Checkpoint dir to start from."}
    )
    logging_dir: Optional[str] = field(
        default=None, metadata={"help": "Logging dir, default will be the run-logs in output_dir."}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=8, metadata={"help": "Batch size of each device."}
    )
    num_train_epochs: Optional[int] = field(
        default=5, metadata={"help": "Number of training epochs."}
    )
    save_total_limit: Optional[int] = field(
        default=2, metadata={"help": "Number of intermediate models to keep."}
    )
    save_steps: Optional[int] = field(
        default=10_000, metadata={"help": "Steps to save between checkpoints."}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Use fp16 training."},
    )


class SynthesisParagraphsDataset(Dataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 training_lmdb: str, block_size: int,
                 batch_size: int = 1):
        assert os.path.isdir(train_data_dir)
        self.block_size = block_size
        self.examples_meta = []
        self.examples: List[Union[bytes, List[int]]] = []
        self.tokenizer = tokenizer
        self.skip = skip

        for path in Path(train_data_dir).glob("*.meta"):
            path = str(path)
            examples_meta, examples = self._load_data(path)
            self.examples_meta.extend(examples_meta)
            self.examples.extend(examples)

        self.skip = self.skip % (len(self.examples) / batch_size)
        if skip:
            logger.info('Skipping %d steps in the current epoch', self.skip)

    def _load_data(self, meta_fn):
        cache_file = meta_fn.rsplit('.', maxsplit=1)[0] + '.cache'

        if not os.path.exists(cache_file):
            self._prepare_cache(meta_fn)

        examples_meta = []
        examples = []
        with open(cache_file, 'rb') as f:
            while True:
                line = f.readline()
                if not line.strip():
                    break
                doi, n_paragraphs = line.strip().split()
                doi = doi.decode()
                n_paragraphs = int(n_paragraphs)
                for i in range(n_paragraphs):
                    examples.append(f.readline())
                examples_meta.append((doi, n_paragraphs))

        logger.info('Loaded %d examples from %s', len(examples), cache_file)
        return examples_meta, examples

    def _process_paper(self, arg):
        doi, paragraphs = arg
        tokenized_paragraphs = []

        for tokenized in map(
                functools.partial(
                    self.tokenizer, add_special_tokens=True,
                    truncation=True,
                    max_length=512), paragraphs):
            if len(tokenized['input_ids']) >= 20:
                tokenized_paragraphs.append(list(tokenized['input_ids']))

        if tokenized_paragraphs:
            return doi, tokenized_paragraphs
        else:
            return None

    def _prepare_cache(self, meta_fn):
        cache_file = meta_fn.rsplit('.', maxsplit=1)[0] + '.cache'
        if os.path.exists(cache_file):
            return

        paragraphs_file = meta_fn.rsplit('.', maxsplit=1)[0] + '.txt'

        total = 0
        with open(meta_fn, encoding='utf8') as f, \
                open(paragraphs_file, encoding='utf8') as p, \
                open(cache_file, 'w') as fout, \
                Pool(processes=10) as pool:
            def iterator_docs():
                for line in f:
                    if not line.strip():
                        continue

                    doi, n_paragraphs = line.strip().split()
                    n_paragraphs = int(n_paragraphs)

                    paragraphs = [p.readline().strip() for _ in range(n_paragraphs)]
                    yield doi, paragraphs

            for i in tqdm(pool.imap(self._process_paper, iterator_docs(), chunksize=5)):
                if i is None:
                    continue
                total += len(i[1])
                fout.write('%s\t%d\n' % (i[0], len(i[1])))
                for j in i[1]:
                    fout.write(' '.join(map(str, j)))
                    fout.write('\n')

        logger.info('Created cache for %d examples to %s', total, meta_fn)

    def __len__(self):
        return len(self.examples)

    zero_tensor = torch.tensor([0], dtype=torch.long)

    def __getitem__(self, i) -> torch.Tensor:
        if self.skip > 0:
            self.skip -= 1
            return self.zero_tensor

        if isinstance(self.examples[i], bytes):
            self.examples[i] = list(map(int, self.examples[i].strip().split()))
        return torch.tensor(self.examples[i], dtype=torch.long)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingOptions))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = BertTokenizerFast.from_pretrained('./synthesis-bert-model', do_lower_case=False)

    config = BertConfig(
        num_hidden_layers=12
    )

    trainer_arguments = TrainingArguments(
        output_dir=training_args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,
        logging_dir=training_args.logging_dir or os.path.join(training_args.output_dir, 'run-logs'),
        fp16=training_args.fp16,
        evaluate_during_training=True,
        eval_steps=1,
    )

    checkpoint_dir = training_args.checkpoint_dir
    skip_samples = 0
    if checkpoint_dir is None:
        checkpoints = [int(str(x).split('-')[-1]) for x in Path(training_args.output_dir).glob("checkpoint-*")]
        if checkpoints:
            skip_samples = max(checkpoints)
            checkpoint_dir = os.path.join(training_args.output_dir, 'checkpoint-%d' % max(checkpoints))

    if checkpoint_dir:
        model = BertForMaskedLM.from_pretrained(checkpoint_dir)
    else:
        model = BertForMaskedLM(config=config)
    dataset = SynthesisParagraphsDataset(
        tokenizer=tokenizer,
        train_data_dir=data_args.train_data_dir,
        block_size=data_args.block_size,
        skip=trainer_arguments.train_batch_size * skip_samples,
        batch_size=trainer_arguments.gradient_accumulation_steps
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=data_args.mlm_probability,
    )

    def save_every_epoch(self: Trainer):
        if hasattr(self, '_last_epoch'):
            if math.floor(self.epoch) != math.floor(self._last_epoch):
                save_path = os.path.join(training_args.output_dir, 'epoch_save_%d' % int(math.floor(self._last_epoch)))
                self.save_model(save_path)

        self._last_epoch = self.epoch

    Trainer.evaluate = save_every_epoch

    trainer = Trainer(
        model=model,
        args=trainer_arguments,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )

    trainer.train(checkpoint_dir)

    trainer.save_model(training_args.output_dir)


if __name__ == '__main__':
    main()
