import os
from argparse import ArgumentParser
from multiprocessing import Semaphore, cpu_count, Process, Queue

import lmdb
import numpy
from tqdm import tqdm
from transformers import BertTokenizerFast

__author__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'
__maintainer__ = 'Haoyan Huo'

__all__ = ['tokenize_lmdb']


def _tokenize_subprocess(tokenizer: BertTokenizerFast, semaphore: Semaphore,
                         documents_queue: Queue, writer_queue: Queue):
    """
    Tokenize paragraphs in a subprocess.

    :param tokenizer: A bert tokenizer.
    :param semaphore: A semaphore to control the throttle of source documents.
    :param documents_queue: A queue from which documents are fetched.
    :param writer_queue: A queue to which tokenized documents are written.
    :return:
    """
    while True:
        item = documents_queue.get()
        semaphore.release()
        if item is None:
            break

        doi, paragraphs = item
        tokenized_paragraphs = []
        for p in paragraphs:
            tokens = tokenizer.tokenize(p, add_special_tokens=True)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            tokenized_paragraphs.append(numpy.array(token_ids))

        writer_queue.put((doi, tokenized_paragraphs))


def _write_db_subprocess(tokenized_lmdb_path: str, writer_queue: Queue):
    dst_env = lmdb.open(
        tokenized_lmdb_path, readonly=False, lock=True, map_size=1024 * 1024 * 1024 * 100)
    dst_txn = dst_env.begin(buffers=True, write=True)

    dst_meta = os.path.join(tokenized_lmdb_path, 'meta.txt')

    with open(dst_meta, 'w') as dst_meta_f:
        while True:
            # Main process sends None to indicate EOF.
            item = writer_queue.get()
            if item is None:
                break

            # Using a consistent dtype long so that trainer can directly use the
            # mapped memory to create arrays.
            doi, tokenized_paragraphs = item
            for i, p in enumerate(tokenized_paragraphs):
                dst_txn.put(f"{doi} {i}".encode('utf8'), p.astype(numpy.long).tobytes())

            # Minus 2 since we have [CLS] and [SEP].
            paragraph_lens = map(str, [x.size - 2 for x in tokenized_paragraphs])
            dst_meta_f.write(f'{doi}\t{",".join(paragraph_lens)}\n')

        dst_txn.commit()
        dst_env.close()


def tokenize_lmdb(
        lmdb_path: str,
        tokenized_lmdb_path: str,
        bert_tokenizer: str,
        cased: bool,
        processes: int = cpu_count()
):
    """
    Tokenize all paragraphs in a LMDB database.

    :param lmdb_path: Source folder of plain text paragraphs.
    :param tokenized_lmdb_path: Destination folder of tokenized paragraphs.
    :param bert_tokenizer: Folder that contains a bert tokenizer.
    :param cased: Whether the tokenizer is cased.
    :param processes: Number of processes to use.
    :return:
    """
    tokenizer = BertTokenizerFast.from_pretrained(
        bert_tokenizer, do_lower_case=not cased)

    src_env = lmdb.open(
        lmdb_path, readonly=True, lock=True)
    src_txn = src_env.begin(buffers=True)

    src_meta = os.path.join(lmdb_path, 'meta.txt')

    semaphore = Semaphore(1000)
    tokenized_queue = Queue()

    def _paragraph_generator():
        with open(src_meta) as src_meta_f:
            for line in src_meta_f:
                # If queue insertion is too fast, we get throttled.
                semaphore.acquire()

                paragraphs = []
                doi, np = line.split('\t')
                for i in range(int(np)):
                    key = f'{doi} {i}'.encode('utf8')
                    paragraph = src_txn.get(key).tobytes().decode('utf8')
                    paragraphs.append(paragraph)
                yield doi, paragraphs

    # Create database writer.
    db_writer = Process(target=_write_db_subprocess, args=(tokenized_lmdb_path, tokenized_queue))
    db_writer.start()

    # Create workers.
    document_queues = [Queue() for _ in range(processes)]
    workers = [Process(
        target=_tokenize_subprocess,
        args=(tokenizer, semaphore, document_queues[i], tokenized_queue)) for i in range(processes)]
    [i.start() for i in workers]

    # Distribute tasks.
    for i, item in enumerate(tqdm(_paragraph_generator(), desc='Tokenizing paragraphs')):
        document_queues[i % len(document_queues)].put(item)

    print('Notifying workers EOF...')
    for queue in document_queues:
        queue.put(None)

    # Wait for workers to finish
    for i, worker in enumerate(workers):
        print(f'Waiting for worker {i} to finish...')
        worker.join()

    print('Notifying database writer EOF...')
    tokenized_queue.put(None)

    # Wait for database write to finish
    print('Waiting for database writer to finish...')
    db_writer.join()


def _main():
    parser = ArgumentParser(description='Tokenize paragraphs in a LMDB paragraphs database.')

    parser.add_argument('--lmdb_path', '-input', type=str, required=True,
                        help='Source folder of the LMDB database.')
    parser.add_argument('--tokenized_lmdb_path', '-output', type=str, required=True,
                        help='Source folder of the LMDB database.')
    parser.add_argument('--tokenizer_path', '-tokenizer', type=str, required=True,
                        help='Folder that contains a BERT tokenizer.')
    parser.add_argument('--cased', '-cased', action='store_true',
                        help='Tokenizer should be case-sensitive.')
    parser.add_argument('--processes', '-p', type=int, default=cpu_count(),
                        help='Tokenizer should be case-sensitive.')

    args = parser.parse_args()

    assert not os.path.exists(args.tokenized_lmdb_path), f"Output dir {args.tokenized_lmdb_path} already exists!"

    tokenize_lmdb(
        lmdb_path=args.lmdb_path,
        tokenized_lmdb_path=args.tokenized_lmdb_path,
        bert_tokenizer=args.tokenizer_path,
        processes=args.processes,
        cased=args.cased,
    )


if __name__ == '__main__':
    _main()
