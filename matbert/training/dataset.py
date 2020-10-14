import logging
import os
from typing import List, Tuple

import lmdb
import numpy
import torch
from torch.utils.data import Dataset

__author__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'
__maintainer__ = 'Haoyan Huo'

__all__ = ['SynthesisParagraphsDataset']


class SynthesisParagraphsDataset(Dataset):
    def __init__(self,
                 training_lmdb: str,
                 skip: int = 0,
                 min_tokens: int = 22,
                 max_tokens: int = 512):
        """
        Constructor.

        :param training_lmdb: Path to a tokenized paragraphs database.
        :param skip: How many samples to skip.
        :param min_tokens: Minimal number of tokens.
        :param max_tokens: Maximal number of tokens to keep.
        """

        self.db_env = lmdb.open(
            training_lmdb, readonly=True, readahead=False, lock=False)
        self.db_txn = self.db_env.begin(buffers=True)

        self.skip = skip
        if skip:
            logging.info('Skipping %d items in the current epoch', self.skip)
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

        meta_fn = os.path.join(training_lmdb, 'meta.txt')
        dois, token_counts = self._load_token_counts(meta_fn)
        self.dois: List[bytes] = dois
        self.token_counts: List[Tuple[int, bytes, int]] = token_counts

        dtype_fn = os.path.join(training_lmdb, 'dtype.txt')
        with open(dtype_fn) as f:
            self.dtype = numpy.dtype(f.read().strip())

    def _load_token_counts(self, meta_fn: str):
        """
        Load token count stats from meta file.
        The meta file will have a format like this:

        10.1000/some-doi   (single TAB)    0:446,1:306,2:0,3:118,4:103,5:120,6:253,7:76,8:76
        """
        dois = []
        token_counts = []
        with open(meta_fn, 'rb') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue

                _doi, _token_counts = line.split(b'\t')

                # Re-order i
                i = len(dois)
                dois.append(_doi)

                for _token_count in _token_counts.split(b','):
                    ip, count = _token_count.split(b':')
                    if count >= self.min_tokens:
                        token_counts.append((i, ip, int(count)))
        return dois, token_counts

    def __len__(self):
        return len(self.token_counts)

    zero_tensor = torch.tensor([0], dtype=torch.long)

    def __getitem__(self, i) -> torch.Tensor:
        if self.skip > 0:
            self.skip -= 1
            return self.zero_tensor

        doi_i, ip, count = self.token_counts[i]
        doi = self.dois[doi_i]
        paragraph_id = doi + b' ' + ip
        paragraph_tokens = self.db_txn.get(paragraph_id)

        paragraph_tokens_array = numpy.frombuffer(paragraph_tokens, dtype=self.dtype).astype(numpy.long)
        if paragraph_tokens_array.size > self.max_tokens:
            # Keep the last element because we need [SEP]
            paragraph_tokens_array = numpy.concatenate(
                (paragraph_tokens_array[:self.max_tokens - 1], paragraph_tokens_array[-1:])
            )
        return torch.tensor(paragraph_tokens_array, dtype=torch.long)
