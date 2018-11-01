"""
    Visualize whitespaces in corpus by replacing it into open box ␣.
    And make a corpus, which is a concated long one sentence.
"""
import os
import argparse
import re
from tqdm import tqdm

def main(config):
    f = open(config.corpus_path, mode='r', encoding=config.encoding, errors='ignore')
    lines = f.readlines()
    f.close()

    """
        All sentences are concatenated and all white spaces are converted to '␣' for visualization.
    """
    f_processed = open(config.processed_path, mode='w', encoding=config.encoding, errors='ignore')

    processed_corpus = ""

    if config.verbose: bar = tqdm(desc='Visualize whitespace in corpus by replacing it into another symbol.', total=len(lines),
        mininterval=1, bar_format='{desc}: {percentage:3.0f}% ({remaining} left)')

    for line in lines:
        if config.verbose: bar.update()

        # remove whitespaces at start and end of line
        line = line.strip()

        # convert other white spaces into space or newline
        line = line.replace('\t'    , ' ' )
        line = line.replace('\x0b'  , '\n')
        line = line.replace('\x0c'  , '\n')
        line = line.replace('\x85'  , ' ' )
        line = line.replace('\xa0'  , ' ' )
        line = line.replace('\u2000', ' ' )
        line = line.replace('\u2001', ' ' )
        line = line.replace('\u2002', ' ' )
        line = line.replace('\u2003', ' ' )
        line = line.replace('\u2004', ' ' )
        line = line.replace('\u2005', ' ' )
        line = line.replace('\u2006', ' ' )
        line = line.replace('\u2007', ' ' )
        line = line.replace('\u2008', ' ' )
        line = line.replace('\u2009', ' ' )
        line = line.replace('\u200a', ' ' )
        line = line.replace('\u2028', ' ' )
        line = line.replace('\u2029', ' ' )
        line = line.replace('\u202f', ' ' )
        line = line.replace('\u205f', ' ' )
        line = line.replace('\u3000', ' ' )

        # replace repetitive space and newline into one
        line = re.sub(' +',' ',line)
        line = re.sub('\n+','\n',line)

        # remove empty sentence and align lines
        for l in line.split('\n'):
            l = l.strip()
            if l != '':
                processed_corpus += (l+'\n')

    # concat lines
    processed_corpus = processed_corpus.strip()
    processed_corpus = processed_corpus.replace(' ', '␣')
    processed_corpus = processed_corpus.replace('\n', '␣')
    processed_corpus = re.sub('␣+','␣',processed_corpus)
    f_processed.write(processed_corpus)
    f_processed.close()

    if config.verbose: bar.close()

    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGNS-WNE/1_preprocess/main.py')
    parser.add_argument('--corpus-path', type=str, default='../data/sample.txt', help='corpus path')
    parser.add_argument('--processed-path', type=str, default='../data/sample_processed.txt', help='output path')
    parser.add_argument('--encoding', type=str, default='utf-8', help='encoding')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    config = parser.parse_args()
    print(config)
    main(config)
