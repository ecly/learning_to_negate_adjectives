"""
Logic for building the data using NLTK and the binary of Google's
300 dimensional word2vec embeddings, trained on Google News data.

The pre-trained embeddings can be downloaded here:
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
"""
import sys
import gensim
import torch
from nltk.corpus import wordnet as wn

from sklearn.neighbors import NearestNeighbors
import numpy as np

CENTROID_MIN_BASIS = 10
ANTONYM_THESAURUS = "./test/thesaurus_antonyms_extended"
LB_GOLD_STANDRD = "./test/lb.inputwords"
GRE_FILTERED_WORDS = "./test/gre_test_adjs_inputwords.txt"
GRE_TEST_QUESTIONS = "./test/gre_testset_adjs.txt"
GOOGLE_NEWS_PATH = "./GoogleNews-vectors-negative300.bin"


class Adjective:
    """
    Class encapsulating an Adjective with its name,
    embedding, hyponyms and antonyms as per WordNet.
    """

    def __init__(self, name, embedding, hyponyms, antonyms):
        self.name = name
        self.embedding = embedding
        self.antonyms = antonyms
        self.hyponyms = hyponyms

    def __str__(self):
        return "Name: %s\nAntonyms: %s\nHyponyms: %s" % (
            self.name,
            self.antonyms,
            self.hyponyms,
        )


class Model:
    def __init__(self, adj2adj):
        self.adj2adj = adj2adj
        self.emb2adj = {a.embedding: a for _, a in adj2adj.items()}
        self.tensors = [a.embedding for _, a in adj2adj.items()]
        self.knn = self.make_knn_from_dict(self.tensors)


    def make_knn_from_dict(self, tensors):
        np_vectors = [t.numpy() for t in tensors]
        knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute").fit(
            np_vectors
        )
        return knn


    def adj_from_name(self, name):
        return self.adj2adj[name]


    def adj_from_vector(self, vector):
        nn = self.knn.kneighbors(
            vector.reshape(1, -1), n_neighbors=1, return_distance=False
        )

        return self.emb2adj[self.tensors[nn[0][0]]]


    def word_from_vector(self, vector):
        return self.adj_from_vector(vector).name


    def kneighbors(self, adj, k):
        nn = self.knn.kneighbors(
            adj.embedding.numpy().reshape(1, -1), n_neighbors=k, return_distance=False
        )
        return list(map(lambda n: self.tensors[n], nn[0]))


def calc_centroid(matrix):
    """
    Calculate centroid of list of torch tensors.

    Returns: 1D torch tensor.
    """

    return torch.mean(matrix, dim=0)


def find_gate_vector(adj, model):
    """
    Finds a gate vector for an adjective using the given model.
    """
    hyp_count = len(adj.hyponyms)
    hyp_emb = list(map(lambda a: model.adj_from_name(a).embedding, adj.hyponyms))
    filter_hyp_emb = set(hyp_emb) | {adj.embedding}

    if hyp_count < CENTROID_MIN_BASIS:
        neighbors = model.kneighbors(adj, CENTROID_MIN_BASIS)
        relevant = [n for n in neighbors if n not in filter_hyp_emb]
        missing_hyp = CENTROID_MIN_BASIS - hyp_count
        hyp_emb = hyp_emb + relevant[:missing_hyp]

    return calc_centroid(torch.stack(hyp_emb))


def build_hyponym_groups():
    """
    Hyponym groups are recognized by their 'lead'
    adjective. The one with synset.pos() = 'a'.
    """
    hyponym_groups = {}

    current_hyponyms = set()
    for synset in wn.all_synsets(wn.ADJ):
        if synset.pos() == "a":
            current_hyponyms = set()
            hyponym_groups[synset.name()] = current_hyponyms

        for word in synset.lemmas():
            current_hyponyms.add(word.name())

    return hyponym_groups


def antonyms_for_synset(synset):
    """Gets all antonyms for a given synset using its lemmas"""
    antonyms = set()
    for word in synset.lemmas():
        for antonym in word.antonyms():
            antonyms.add(antonym.name())

    return antonyms


def build_adjective_dict(model):
    """
    Build adjective dict using wordnet and the given
    model for adjective word embeddings
    """
    word2adj = {}
    hyponym_groups = build_hyponym_groups()

    current_hyponyms = set()
    current_antonyms = set()
    for synset in wn.all_synsets(wn.ADJ):
        if synset.pos() == "a":
            current_hyponyms = hyponym_groups[synset.name()]
            current_antonyms = antonyms_for_synset(synset)

        for word in synset.lemmas():
            word_name = word.name()

            if word_name not in word2adj:
                try:
                    embedding = torch.from_numpy(model.get_vector(word_name))
                    word2adj[word_name] = Adjective(word_name, embedding, set(), set())
                except KeyError:
                    continue

            adj = word2adj[word_name]
            adj.antonyms = adj.antonyms | current_antonyms
            adj.hyponyms = adj.hyponyms | current_hyponyms

    return word2adj


def build_training_pairs(model, filtered=None):
    """
    Builds a list of <adjective, cohyponym, antonym> triples
    for the given model. The model contains all the adjectives,
    and allows querying for embeddings using antonym names.

    Optionally takes an enumerable of filtered words from
    which we filter pairs where the input adjective is in that
    enumerable.
    """
    filtered = [] if filtered is None else filtered
    pairs = []
    for adj in model.adj2adj.values():
        for adj_name in adj.hyponyms | {adj.name}:
            if adj_name in filtered:
                continue
            for ant_name in adj.antonyms:
                try:
                    current_adj = model.adj_from_name(adj_name)
                    adj_emb = current_adj.embedding
                    ant_emb = model.adj_from_name(ant_name).embedding
                    centroid = find_gate_vector(current_adj, model)
                    pairs.append((adj_emb, centroid, ant_emb))
                except KeyError:
                    # print("failed for %s and %s" % (adj.name, ant))
                    continue

    return pairs


def load_gre_filtered_words():
    """
    Loads and creates a set of the input words
    for the GRE test set.
    """
    with open(GRE_FILTERED_WORDS, "r") as f:
        words = set()
        for word in f:
            words.add(word.strip().lower())

        return words


def load_gre_test_set():
    """
    Loads and creates a test set of tuples
    <input, [options], answer> for antonym prediction.
    """
    with open(GRE_TEST_QUESTIONS, "r") as f:
        test_data = []
        for line in f:
            adj, rest = line.split(": ", 1)
            options, answer = rest.split(" :: ")
            test_data.append((adj, options.strip().split(), answer.strip()))

        return test_data


def load_gold_standard():
    """
    Loads and creates a dictionary mapping the 99 LB adjectives
    to a list of their 'gold standard' antonyms.
    """
    data = {}
    with open(LB_GOLD_STANDRD, "r") as f:
        for word in f:
            data[word.strip()] = []

    with open(ANTONYM_THESAURUS, "r") as f:
        for line in f:
            adj, ant = line.strip().split(" ", 1)
            if adj in data:
                data[adj].append(ant)

    return data


def main():
    # Load the Google news pre-trained Word2Vec model
    gensim_model = gensim.models.KeyedVectors.load_word2vec_format(
        GOOGLE_NEWS_PATH, binary=True
    )
    adj2adj = build_adjective_dict(gensim_model)
    model = Model(adj2adj)
    filtered_words = load_gre_filtered_words()
    pairs = build_training_pairs(model, filtered_words)
    readable_pairs = list(
        map(
            lambda x: (model.word_from_vector(x[0]),
                       model.word_from_vector(x[1]),
                       model.word_from_vector(x[2])),
            pairs,
        )
    )
    for p in readable_pairs:
        print(p)

    print(len(pairs))


if __name__ == "__main__":
    main()
