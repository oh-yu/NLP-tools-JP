import gzip
import re

from gensim.models.word2vec import Word2Vec
import MeCab
import numpy as np
import ot
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

WORD2VEC_PATH = './word2vec.gensim.model'
REPLACE_WORDS = [chr(i) for i in range(65, 91)]
# ["A", "B", "C", ... , "Z"]
UNKNOWN_TO_REPLACEMENT = {}
REPLACE_COUNT = 0


def _tokens_to_vectors(sentence, word2vec):
    global REPLACE_COUNT
    vectors = []
    for word in sentence:
        try:
            vectors.append(list(word2vec.wv[word]))
        except KeyError:
            if word not in UNKNOWN_TO_REPLACEMENT.keys():
                replace_word = REPLACE_WORDS[REPLACE_COUNT]
                vectors.append(list(word2vec.wv[replace_word]))

                UNKNOWN_TO_REPLACEMENT[word] = replace_word
                REPLACE_COUNT += 1
            else:
                vectors.append(list(word2vec.wv[UNKNOWN_TO_REPLACEMENT[word]]))
    return vectors


def _sentences_to_vectors(sentence1, sentence2, word2vec):
    vectors1 = _tokens_to_vectors(sentence1, word2vec)
    vectors2 = _tokens_to_vectors(sentence2, word2vec)
    return vectors1, vectors2


def word_movers_dist(sentence1: str, sentence2: str) -> float:
    """
    https://proceedings.mlr.press/v37/kusnerb15.html

    Min: Σ_{i, j} c_ij * x_ij
    s.t. 
    (1) Σ_i x_ij = a_j, j∈{0, 1, 2, ..., M}
    (2) Σ_j x_ij = b_i, i∈{0, 1, 2, ..., N}

    x_ij: amount of flow from i to j
    c_ij: cost(=distance between vectors) of flow from i to j
    a_j: demand of j(uniform)
    b_i: supply of i(uniform)
    """
    # Sentences to Vectors
    tagger = MeCab.Tagger("-Owakati")
    sentence1 = tagger.parse(sentence1).split()
    sentence2 = tagger.parse(sentence2).split()
    word2vec = Word2Vec.load(WORD2VEC_PATH)
    vectors1, vectors2 = _sentences_to_vectors(sentence1, sentence2, word2vec)

    # Solve Optimal Transport
    supply = np.ones(len(vectors1)) / len(vectors1)
    demand = np.ones(len(vectors2)) / len(vectors2)
    cost = pairwise_distances(vectors1, vectors2, metric="euclidean")
    return ot.emd2(supply, demand, cost)


def _extract_verb_and_noun(sentence: str) -> list[str]:
    tagger = MeCab.Tagger()
    parse = tagger.parse(sentence)
    words = []
    lines = parse.split("\n")
    for line in lines:
        items = re.split('[\t,]',line)
        if len(items) >= 2 and items[1] == "名詞" and items[2] != "非自立":
            words.append(items[0])
        elif len(items) >= 2 and items[1] == "動詞":
            words.append(items[0])
    return words


def word_movers_dist_only_verb_and_noun(sentence1: str, sentence2: str) -> float:
    """
    Word Mover's Dist using only verb and noun.
    Same algo as word_movers_dist.
    """
    # Sentences to Vectors
    sentence1 = _extract_verb_and_noun(sentence1)
    sentence2 = _extract_verb_and_noun(sentence2)
    word2vec = Word2Vec.load(WORD2VEC_PATH)
    vectors1, vectors2 = _sentences_to_vectors(sentence1, sentence2, word2vec)

    # Solve Optimal Transport
    supply = np.ones(len(vectors1)) / len(vectors1)
    demand = np.ones(len(vectors2)) / len(vectors2)
    cost = pairwise_distances(vectors1, vectors2, metric="euclidean")
    return ot.emd2(supply, demand, cost)


def word_rotators_dist(sentence1: str, sentence2: str) -> float:
    """
    Almost same algo as word_movers_dist, a_j and b_i as the length of vector, 
    c_ij as cosine similarity. The length of given word vector ∝ the importance of word experimentally.
    Experimental performance is better than word_movers_dist.

    https://arxiv.org/abs/2004.15003
    """
    # Sentences to Vectors
    sentence1 = _extract_verb_and_noun(sentence1)
    sentence2 = _extract_verb_and_noun(sentence2)
    word2vec = Word2Vec.load(WORD2VEC_PATH)
    vectors1, vectors2 = _sentences_to_vectors(sentence1, sentence2, word2vec)

    # Solve Optimal Transport
    supply = np.array([np.linalg.norm(v) for v in vectors1])
    supply /= supply.sum()
    demand = np.array([np.linalg.norm(v) for v in vectors2])
    demand /= demand.sum()
    cost = 1 - cosine_similarity(vectors1, vectors2)
    return ot.emd2(supply, demand, cost)