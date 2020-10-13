import html
import os
from argparse import ArgumentParser

import lmdb
from pymongo import MongoClient
from tqdm import tqdm

__author__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'
__maintainer__ = 'Haoyan Huo'

__all__ = ['collect_paragraphs']


def collect_paragraphs(
        db_hostname: str,
        db_username: str,
        db_password: str,
        lmdb_path: str,
        num_papers: int,
):
    """
    Collect papers from synthesisproject.lbl.gov and save them into LMDB database
    for later training.

    :param db_hostname: Hostname of the database.
    :param db_username: Username for authentication.
    :param db_password: Password for authentication.
    :param lmdb_path: Path to a LMDB database instance.
    :param num_papers: Number of papers to collect.
    :return: None
    """
    assert not os.path.exists(lmdb_path), "Target LMDB path already exists! Delete it first."

    # Open connections to synthesisproject.lbl.gov
    # Full text and paragraphs are saved separately.
    print('Connecting to database...')
    full_text_db = MongoClient(db_hostname).FullText
    full_text_db.authenticate(db_username, db_password)
    print('Connected to FullText database.')
    paper_meta = full_text_db.Paper_Metadata
    prod_db = MongoClient(db_hostname).SynPro
    prod_db.authenticate(db_username, db_password)
    all_paragraphs = prod_db.Paragraphs
    print('Connected to SynPro database.')

    # Retrieve 1.5 more papers just in case there are empty papers
    num_req_papers = int(num_papers * 1.5)
    print(f'Requesting {num_req_papers} paper metadata...')
    papers = list(paper_meta.aggregate(
        [{'$sample': {'size': num_req_papers}}], allowDiskUse=True
    ))

    # For safety, just map 100GB (this is the current upper bound of the size of the
    # synthesisproject text database.
    env = lmdb.open(
        lmdb_path,
        readonly=False, map_size=1024 * 1024 * 1024 * 100, lock=True)
    txn = env.begin(buffers=True, write=True)

    with tqdm(total=num_papers, desc='Downloading papers') as bar, \
            open(os.path.join(lmdb_path, 'meta.txt'), 'w') as meta_f:
        for paper in papers:
            # We don't filter bad paragraphs (e.g. equations) since this
            # should be preferably done by training programs.
            final_paragraphs = []
            for paragraph in all_paragraphs.find({'DOI': paper['DOI']}):
                # If multiple paragraphs accidentally are in one string, we treat
                # them as a single paragraph.
                paragraph = html.unescape(paragraph['text']).replace('\n', ' ').strip()
                final_paragraphs.append(paragraph)

            # Only collect papers that have at least one paragraph!
            if not len(final_paragraphs):
                continue

            # In LMDB, entries are stored as (b"10.0000/test:10", b"Test paragraph...")
            for i, paragraph in enumerate(final_paragraphs):
                key = f'{paper["DOI"]} {i}'
                content = paragraph.encode('utf8', errors='ignore')
                txn.put(key.encode('utf8'), content)

            # Keep a reference in the meta file since LMDB is unordered.
            meta_f.write('%s\t%d\n' % (paper['DOI'], len(final_paragraphs)))

            bar.update(1)
            # We actually collected more than enough papers. Here we should exit
            # early when the number of papers is more than or equal to what is desired.
            if bar.n >= num_papers:
                break

    txn.commit()
    env.close()


def _main():
    parser = ArgumentParser(description='Collect paragraphs for training MatBERT.')

    parser.add_argument('--db_hostname', '-host', type=str, required=True,
                        help='Hostname of synthesisproject.lbl.gov')
    parser.add_argument('--db_username', '-user', type=str, required=True,
                        help='Username of your credentials for accessing the database.')
    parser.add_argument('--db_password', '-pass', type=str, required=True,
                        help='Password of your credentials for accessing the database.')
    parser.add_argument('--lmdb_path', '-lmdb', type=str, required=True,
                        help='Destination folder for storing LMDB database.')
    parser.add_argument('--num_papers', '-num', type=int, default=2 * 1000 * 1000,
                        help='How many papers to collect from the database.')

    args = parser.parse_args()
    collect_paragraphs(
        args.db_hostname, args.db_username, args.db_password,
        args.lmdb_path, args.num_papers)


if __name__ == '__main__':
    _main()
