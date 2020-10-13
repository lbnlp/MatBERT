import os
from argparse import ArgumentParser
from collections import defaultdict
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

        key, paragraph = item
        tokens = tokenizer.tokenize(paragraph.decode('utf8'), add_special_tokens=True)
        token_ids = numpy.array(tokenizer.convert_tokens_to_ids(tokens))

        writer_queue.put((key, token_ids))


def _write_db_subprocess(tokenized_lmdb_path: str, writer_queue: Queue, dtype: numpy.dtype):
    dst_env = lmdb.open(
        tokenized_lmdb_path, readonly=False, lock=True, map_size=1024 * 1024 * 1024 * 100)
    dst_txn = dst_env.begin(buffers=True, write=True)

    dst_meta = os.path.join(tokenized_lmdb_path, 'meta.txt')

    meta_maps = defaultdict(dict)

    while True:
        # Main process sends None to indicate EOF.
        item = writer_queue.get()
        if item is None:
            break

        # Using a consistent dtype long so that trainer can directly use the
        # mapped memory to create arrays.
        key, token_ids = item
        doi, ip = key.split(b' ')

        dst_txn.put(key, token_ids.astype(dtype).tobytes())

        # Minus 2 since we have [CLS] and [SEP].
        meta_maps[doi][int(ip)] = token_ids.size - 2

    dst_txn.commit()
    dst_env.close()

    with open(dst_meta, 'wb') as dst_meta_f:
        for doi, token_counts in meta_maps.items():
            token_count_s = b','.join(
                f'{ip}:{c}'.encode('utf8')
                for ip, c in sorted(token_counts.items()))
            dst_meta_f.write(doi)
            dst_meta_f.write(b'\t')
            dst_meta_f.write(token_count_s)
            dst_meta_f.write(b'\n')


def tokenize_lmdb(
        lmdb_path: str,
        tokenized_lmdb_path: str,
        bert_tokenizer: str,
        cased: bool,
        processes: int = cpu_count(),
        dtype: str = 'uint16'
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
        lmdb_path, readonly=True, lock=False)
    src_txn = src_env.begin(buffers=True)

    semaphore = Semaphore(4096)
    tokenized_queue = Queue()

    def _paragraph_generator():
        for key, value in iter(src_txn.cursor()):
            # If queue insertion is too fast, we get throttled.
            semaphore.acquire()

            yield key.tobytes(), value.tobytes()

    # Create database writer.
    db_writer = Process(
        target=_write_db_subprocess,
        args=(tokenized_lmdb_path, tokenized_queue, numpy.dtype(dtype)))
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
    parser.add_argument('--dtype', '-dtype', type=str, default='uint16',
                        help='Dtype of the stored numpy arrays.')

    args = parser.parse_args()

    assert not os.path.exists(args.tokenized_lmdb_path), f"Output dir {args.tokenized_lmdb_path} already exists!"

    tokenize_lmdb(
        lmdb_path=args.lmdb_path,
        tokenized_lmdb_path=args.tokenized_lmdb_path,
        bert_tokenizer=args.tokenizer_path,
        processes=args.processes,
        cased=args.cased,
        dtype=args.dtype,
    )


if __name__ == '__main__':
    _main()
