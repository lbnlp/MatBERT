import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from transformers import (
    BertConfig, BertTokenizerFast, BertForMaskedLM,
    DataCollatorForLanguageModeling, HfArgumentParser)
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
