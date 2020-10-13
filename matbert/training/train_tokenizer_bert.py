import os
from argparse import ArgumentParser

import lmdb
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm

__author__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'
__maintainer__ = 'Haoyan Huo'

__all__ = ['prepare_for_tokenizer']


def prepare_for_tokenizer(
        lmdb_path: str,
):
    """
    Convert a LMDB-stored paragraphs set into a plain text for use with tokenizer.
    The plain text file will be removed if any exception occurs while creating it.
    """
    pt_filename = os.path.join(lmdb_path, 'all_paragraphs.txt')
    assert not os.path.exists(pt_filename), f"Plain text file {pt_filename} already exists! Delete it first."

    env = lmdb.open(
        lmdb_path, readonly=True, lock=True)
    txn = env.begin(buffers=True)

    try:
        with tqdm(desc='Dumping paragraphs', unit='B', unit_scale=True) as bar, \
                open(pt_filename, 'wb') as meta_f:
            for key, value in iter(txn.cursor()):
                meta_f.write(value)
                bar.update(len(value) + 1)
        env.close()
    except:
        if os.path.exists(pt_filename):
            os.remove(pt_filename)
        raise

    return pt_filename


def _main():
    parser = ArgumentParser(description='Train a MatBERT tokenizer.')

    parser.add_argument('--lmdb_path', '-lmdb', type=str, required=True,
                        help='Source folder of the LMDB database.')
    parser.add_argument('--dont_remove_file', '-no_remove', action='store_true',
                        help='Remove the temporary file containing all paragraphs.')
    parser.add_argument('--save_dir', '-save', type=str, required=True,
                        help='Destination folder of the tokenizer.')

    parser.add_argument('--vocab_size', '-size', type=int, default=30_522,
                        help='Size of the vocabulary.')
    parser.add_argument('--cased', '-cased', action='store_true',
                        help='Tokenizer should be case-sensitive.')

    args = parser.parse_args()

    assert not os.path.exists(args.save_dir), f"Output dir {args.save_dir} already exists!"
    os.makedirs(args.save_dir)

    pt_filename = prepare_for_tokenizer(args.lmdb_path)
    try:
        print(f'Creating new tokenizer to "{os.path.realpath(args.save_dir)}". '
              f'Configs: cased={args.cased}, vocab_size={args.vocab_size}')
        tokenizer = BertWordPieceTokenizer(lowercase=not args.cased)
        tokenizer.train(files=pt_filename, vocab_size=args.vocab_size)
        tokenizer.save_model(args.save_dir)
    finally:
        if not args.dont_remove_file:
            os.remove(pt_filename)


if __name__ == '__main__':
    _main()
