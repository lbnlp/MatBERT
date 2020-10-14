import json
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser

__author__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'
__maintainer__ = 'Haoyan Huo'

__all__ = ['DataOpts', 'ModelOpts', 'TrainingOpts', 'parse_with_config']


@dataclass
class DataOpts:
    tokenizer_path: str = field(
        default=None,
        metadata={"help": "Path to a pretrained BERT tokenizer."}
    )
    cased: bool = field(
        default=None,
        metadata={"help": "Whether the tokenizer is cased."}
    )
    tokenized_lmdb_path: str = field(
        default=None,
        metadata={"help": "The training database containing tokenized paragraphs."}
    )


@dataclass
class ModelOpts:
    output_dir: str = field(
        default=None,
        metadata={"help": "Model output dir."}
    )
    logging_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Logging dir, default will be the run-logs in output_dir."}
    )
    bert_config: dict = field(
        default_factory=dict,
        metadata={"help": "BERT config. Use a json file to specify a dict."}
    )
    load_checkpoint: bool = field(
        default=True,
        metadata={"help": "Load the latest checkpoint in output dir."}
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Number of intermediate models to keep. Default is to keep all."}
    )
    save_steps: Optional[int] = field(
        default=10_000,
        metadata={"help": "Steps to save between checkpoints."}
    )


@dataclass
class TrainingOpts(ModelOpts, DataOpts):
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss."}
    )

    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size of each device."}
    )
    num_train_epochs: int = field(
        default=5,
        metadata={"help": "Number of training epochs."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of steps between gradient accumulation steps."}
    )


@dataclass
class MiscOpts:
    config: Optional[str] = field(
        default=None,
        metadata={"help": "Filename to a config json file."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Use fp16 training."},
    )


def parse_with_config() -> TrainingOpts:
    parser = HfArgumentParser(TrainingOpts)
    opts = vars(parser.parse_args())

    if opts['config'] is not None:
        with open(opts['config']) as f:
            config = json.load(f)
            for key, value in config.items():
                opts[key] = value

    return TrainingOpts(**opts)
