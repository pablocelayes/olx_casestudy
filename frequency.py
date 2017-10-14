#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from corpus import CorpusManager
from corpus.tokenizer import tokenize


def get_tf_corpus():
    queries, _, _, _ = CorpusManager().load_dataset()
    tf_queries_corpus = TfidfVectorizer(
        use_idf=False,
        tokenizer=tokenize).fit_transform(queries)
    return tf_queries_corpus


def save_tf_corpus(tf_queries_corpus):
    with open('tf_corpus.pickle', 'wb') as f:
        pickle.dump(tf_queries_corpus, f)


if __name__ == "__main__":
    tf_queries_corpus = get_tf_corpus()
    print(tf_queries_corpus)
    save_tf_corpus(tf_queries_corpus)
