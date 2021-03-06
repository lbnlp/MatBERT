import logging
import math
import os
from pathlib import Path
from typing import Tuple

import torch.distributed as dist
from transformers import (
    BertConfig, BertTokenizerFast, BertForMaskedLM,
    DataCollatorForLanguageModeling)
from transformers import Trainer, TrainingArguments

from matbert.training.dataset import SynthesisParagraphsDataset
from matbert.training.options import parse_with_config, BertOpts

__author__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'
__maintainer__ = 'Haoyan Huo'

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s][%(name)s]: %(message)s",
    datefmt="%m/%d %H:%M:%S",
    level=logging.INFO,
)


class MyTrainer(Trainer):
    # Hack function used to save models at the end of every epoch.
    def evaluate(self, eval_dataset=None):
        if hasattr(self, '_last_epoch'):
            if math.floor(self.epoch) != math.floor(self._last_epoch):
                save_path = os.path.join(self.args.output_dir, 'epoch_save_%d' % int(math.floor(self._last_epoch)))
                self.save_model(save_path)

        setattr(self, '_last_epoch', self.epoch)


def prepare_model(opts: BertOpts) -> Tuple[BertForMaskedLM, str, int]:
    config = BertConfig(**opts.bert_config)

    checkpoints = list(map(str, Path(opts.output_dir).glob("checkpoint-*")))

    if opts.load_checkpoint and len(checkpoints) > 0:
        checkpoint_steps = [int(x.rsplit('-')[1]) for x in checkpoints]
        latest_global_step = max(checkpoint_steps)
        checkpoint_dir = os.path.join(
            opts.output_dir, 'checkpoint-%d' % latest_global_step)
        model = BertForMaskedLM.from_pretrained(checkpoint_dir, config=config)
    else:
        checkpoint_dir = None
        latest_global_step = 0
        model = BertForMaskedLM(config=config)

    return model, checkpoint_dir, latest_global_step


def main():
    opts = parse_with_config()

    assert os.path.exists(opts.tokenizer_path), \
        "Tokenizer (--tokenizer_path) not specified or does not exist!"
    assert os.path.exists(opts.tokenized_lmdb_path), \
        "Training data (--tokenized_lmdb_path) not specified or does not exist!"
    assert opts.output_dir is not None, \
        "Output dir (--output_dir) is None!"

    tokenizer = BertTokenizerFast.from_pretrained(
        opts.tokenizer_path, do_lower_case=not opts.cased)

    trainer_arguments = TrainingArguments(
        output_dir=opts.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=opts.num_train_epochs,
        gradient_accumulation_steps=opts.gradient_accumulation_steps,
        per_device_train_batch_size=opts.per_device_train_batch_size,
        weight_decay=opts.weight_decay,
        dataloader_drop_last=opts.dataloader_drop_last,
        save_steps=opts.save_steps,
        save_total_limit=opts.save_total_limit,
        logging_dir=opts.logging_dir or os.path.join(opts.output_dir, 'run-logs'),
        dataloader_num_workers=4,
        fp16=opts.fp16,
        fp16_opt_level=opts.fp16_opt_level,
        seed=opts.seed,
        local_rank=opts.local_rank,

        # A nice hack for storing models every epoch. See below
        evaluate_during_training=True,
        eval_steps=1,
    )

    model, checkpoint_dir, latest_global_step = prepare_model(opts)
    logger.info('Loaded model from %s', checkpoint_dir)

    dataset = SynthesisParagraphsDataset(
        training_lmdb=opts.tokenized_lmdb_path,
        max_tokens=model.config.max_position_embeddings
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=opts.mlm_probability,
    )

    # Recover how many samples to skip
    # Need to call trainer_arguments.device to setup pytorch.
    _ = trainer_arguments.device
    if opts.local_rank > -1:
        num_replicas = dist.get_world_size()
        local_data_size = int(math.ceil(len(dataset) * 1.0 / num_replicas))
    else:
        local_data_size = len(dataset)
    local_batches = local_data_size // opts.per_device_train_batch_size
    steps_trained_in_current_epoch = latest_global_step % (
            local_batches // opts.gradient_accumulation_steps
    )
    dataset.skip = steps_trained_in_current_epoch * \
                   opts.gradient_accumulation_steps * \
                   opts.per_device_train_batch_size

    trainer = MyTrainer(
        model=model,
        args=trainer_arguments,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )

    trainer.train(checkpoint_dir)
    trainer.save_model(opts.output_dir)


if __name__ == '__main__':
    main()
