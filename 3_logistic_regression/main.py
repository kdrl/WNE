import os
import re
import math
import random
import argparse
import numpy as np
import pandas as pd
import h5py
import locale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
from tqdm import tqdm

locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' )

def stochastic_word_segmentation(args):
    global processed_corpus, config
    process_num = args[0]
    indexes = args[1]
    #print("Process {} : processing corpus from {}-th letter to {}-th letter".format(process_num, indexes[0], indexes[-1]))
    verbose = False
    if process_num == 0:
        verbose = True
    if verbose:
        bar = tqdm(desc='Get word boundary', total=len(indexes), mininterval=1, bar_format='{desc}: {percentage:3.0f}% ({remaining} left)')
    X_each = np.empty((len(indexes), config.max_n**2))
    for each_i,i in enumerate(indexes):
        if verbose: bar.update()
        explanatory_variables=list()
        for a in range(1,config.max_n+1):
            for b in range(1,config.max_n+1):
                explanatory_variables.append(get_association(processed_corpus[i-a:i], processed_corpus[i:i+b]))
        X_each[each_i] = np.array(explanatory_variables)
    if verbose: bar.close()
    return X_each

def get_association(a, b):
    global ngram_occurence, corpus_length
    return math.log( (ngram_occurence.get(a+b, 1) * corpus_length) / (ngram_occurence.get(a, 1) * ngram_occurence.get(b, 1)) )

def get_ngram_occurence(path, verbose=True):
    f = open(path)
    lines = f.readlines()
    f.close()
    ngram_occurence = dict()
    if verbose:
        bar = tqdm(desc='Get frequency of n-gram', total=len(lines), mininterval=1, bar_format='{desc}: {percentage:3.0f}% ({remaining} left)')
    for line in lines:
        if verbose: bar.update()
        ngram, count = line.split()
        count = locale.atoi(count)
        ngram_occurence[ngram] = count
    if verbose: bar.close()
    return ngram_occurence

def clean(line):
    # visualize whitespaces in the same way as 1_preprocess/main.py
    line = line.replace(' '     , '␣' )
    line = line.replace('\n'    , '␣' )
    line = line.replace('\t'    , '␣' )
    line = line.replace('\n'    , '↵' )
    line = line.replace('\x0b'  , '␣' )
    line = line.replace('\x0c'  , '␣' )
    line = line.replace('\x85'  , '␣' )
    line = line.replace('\xa0'  , '␣' )
    line = line.replace('\u2000', '␣' )
    line = line.replace('\u2001', '␣' )
    line = line.replace('\u2002', '␣' )
    line = line.replace('\u2003', '␣' )
    line = line.replace('\u2004', '␣' )
    line = line.replace('\u2005', '␣' )
    line = line.replace('\u2006', '␣' )
    line = line.replace('\u2007', '␣' )
    line = line.replace('\u2008', '␣' )
    line = line.replace('\u2009', '␣' )
    line = line.replace('\u200a', '␣' )
    line = line.replace('\u2028', '␣' )
    line = line.replace('\u2029', '␣' )
    line = line.replace('\u202f', '␣' )
    line = line.replace('\u205f', '␣' )
    line = line.replace('\u3000', '␣' )
    line = re.sub('␣+','␣',line)
    return line

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Probabilistic predictor of word boundary for SGNS-WNE')
    parser.add_argument('--corpus-path', type=str, default="../data/sample.txt")
    parser.add_argument('--segmented-corpus-path', type=str, default="../data/sample_segmented.txt")
    parser.add_argument('--ngram-count-path', type=str, default="../data/ngram_frequency.csv")
    parser.add_argument('--processed-corpus-path', type=str, default="../data/sample_processed.txt")
    parser.add_argument('--word-boundary-path', type=str, default="../data/word_boundary.hdf5")
    parser.add_argument('--usage-ratio', type=float, default=0.1)
    parser.add_argument('--random-seed', type=int, default=2018)
    parser.add_argument('--max-n', type=int, default=4)
    parser.add_argument('--n_core', type=int, default=8)
    config = parser.parse_args()
    print(config.__dict__)
    np.random.seed(config.random_seed)

    # prepare data for training predictor
    f1 = open(config.corpus_path, "r")
    f2 = open(config.segmented_corpus_path, "r")
    sentences = f1.readlines()
    segmented_sentences = f2.readlines()
    f1.close()
    f2.close()
    assert (len(sentences) == len(segmented_sentences))

    concat_sentence = str()
    labels = list()

    # randomly use the part of corpus
    usage_sentence_num = int(len(sentences)*config.usage_ratio)
    index = np.random.randint(0,len(sentences)-usage_sentence_num)
    sentences = sentences[index:index+usage_sentence_num]
    segmented_sentences = segmented_sentences[index:index+usage_sentence_num]
    print("Train predictor with {}% of the corpus ({} sentences)".format(int(config.usage_ratio*100), len(sentences)))
    for i,sentence in enumerate(sentences):
        sentence = clean(sentence.strip()+'␣')
        concat_sentence += sentence
        segmented_sentence = clean(segmented_sentences[i].strip()+'␣')
        assert(sentence[0] == segmented_sentence[0])
        assert(len(segmented_sentence) >= len(sentence))
        gap = 0
        is_new_word = 1
        label = list()
        for j,character in enumerate(sentence):
            character_in_segmented_sentence= segmented_sentence[j+gap]
            if character == '␣':
                is_new_word = 1
                label.append(is_new_word)
                if character != character_in_segmented_sentence:
                    gap -= 1
                continue
            while(character != character_in_segmented_sentence):
                gap += 1
                is_new_word = 1
                character_in_segmented_sentence = segmented_sentence[j+gap]
            label.append(is_new_word)
            is_new_word = 0
        labels.extend(label)

    # check segmentation labeling
    print("Check segmentation labeling")
    assert(len(labels) == len(concat_sentence))
    print(sentences[0])
    print(segmented_sentences[0])
    print()
    print(concat_sentence[:10])
    print(labels[:10])
    print()

    # get processed_corpus here
    processed_corpus = open(config.processed_corpus_path, 'r')
    lines = processed_corpus.readlines()
    processed_corpus.close()
    assert(len(lines) == 1)
    processed_corpus = lines[0]
    corpus_length = len(processed_corpus)

    # get ngram frequency
    ngram_occurence = get_ngram_occurence(config.ngram_count_path)

    # build X, Y for predictor
    Y = np.array(labels)[config.max_n:-(config.max_n-1)]
    X = np.empty((len(concat_sentence)-config.max_n-(config.max_n-1), config.max_n**2))
    assert(X.shape[0] == Y.shape[0])
    for i in range(config.max_n,len(concat_sentence)-(config.max_n-1)):
        explanatory_variables=list()
        for a in range(1,config.max_n+1):
            for b in range(1,config.max_n+1):
                explanatory_variables.append(get_association(concat_sentence[i-a:i], concat_sentence[i:i+b]))
        X[i-config.max_n] = np.array(explanatory_variables)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)

    # build and train
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # # check
    # print("Check the performance of predictor of word boundary")
    # print(model.coef_)
    # print(model.intercept_)
    # print(model.get_params())
    # print(model.densify())
    # print(model.score(X_test, Y_test))
    # print(model.predict_proba(X_test))

    # predict word boundary
    print("Prediction of word boundary starts (corpus length:{})".format(len(processed_corpus)))
    word_boundary = np.array([1])
    p = Pool(config.n_core)
    chunk_size = (int)(((len(processed_corpus)-1)/config.n_core) + 1)
    output = p.map(stochastic_word_segmentation, [(process_num, range(len(processed_corpus))[i:i+chunk_size]) for process_num,i in enumerate(range(1,len(processed_corpus),chunk_size))])
    X = np.vstack(output)
    word_boundary = np.hstack((word_boundary,model.predict_proba(X)[:,1]))
    assert(word_boundary.shape[0] == len(processed_corpus))

    # check
    print("Calculation done. Word boundary samples from the head : ", word_boundary[:10])

    # save
    f = h5py.File(config.word_boundary_path, "w")
    signal_ds = f.create_dataset("word_boundary", data=word_boundary)
    f.close()
