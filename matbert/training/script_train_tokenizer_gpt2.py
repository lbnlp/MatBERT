import os
from argparse import ArgumentParser

from tokenizers import ByteLevelBPETokenizer

__author__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'
__maintainer__ = 'Haoyan Huo'

from matbert.training.script_train_tokenizer_bert import prepare_for_tokenizer


def _main():
    parser = ArgumentParser(description='Train a MatGPT2 tokenizer.')

    parser.add_argument('--lmdb_path', '-lmdb', type=str, required=True,
                        help='Source folder of the LMDB database.')
    parser.add_argument('--dont_remove_file', '-no_remove', action='store_true',
                        help='Do not remove the temporary file containing all paragraphs.')
    parser.add_argument('--save_dir', '-save', type=str, required=True,
                        help='Destination folder of the tokenizer.')

    parser.add_argument('--vocab_size', '-size', type=int, default=50_257,
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
        tokenizer = ByteLevelBPETokenizer(lowercase=not args.cased)
        tokenizer.train(files=pt_filename, vocab_size=args.vocab_size, special_tokens=['<|endoftext|>'])
        tokenizer.save_model(args.save_dir)
    finally:
        if not args.dont_remove_file:
            os.remove(pt_filename)


if __name__ == '__main__':
    _main()
