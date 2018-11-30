"""
Logic for building the data using NLTK and the binary of Google's
300 dimensional word2vec embeddings, trained on Google News data.

The pre-trained embeddings can be downloaded here:
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
"""
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
ADJ2EBM_PATH = "adj_emb.tsv"


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

class AdjectiveModel:
    """
    An Adjective Model encapsulating mappings from
    embeddings to adjectives and names to adjectives.

    Also manages querying for k nearest neighbors among
    all known adjectives.
    """
    def __init__(self, adj2adj):
        self.adj2adj = adj2adj
        self.emb2adj = {a.embedding: a for a in adj2adj.values()}
        self.tensors = [a.embedding for a in adj2adj.values()]
        self.knn = AdjectiveModel.make_knn_from_dict(self.tensors)

    @staticmethod
    def make_knn_from_dict(tensors):
        """Make knn model using sklearn"""
        np_vectors = [t.numpy() for t in tensors]
        knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="auto").fit(
            np_vectors
        )
        return knn

    def adj_from_name(self, name):
        """Get an Adjective from its name"""
        return self.adj2adj[name]

    def adj_from_vector(self, vector):
        """Get an Adjective from pytorch tensor embedding"""
        neighbors = self.knn.kneighbors(
            vector.reshape(1, -1), n_neighbors=1, return_distance=False
        )

        return self.emb2adj[self.tensors[neighbors[0][0]]]

    def word_from_vector(self, vector):
        """Get name of the Adjective with embedding closest to given vector"""
        return self.adj_from_vector(vector).name

    def kneighbors(self, adj, k):
        """Get the tensors of the k-nearest neighbors to given Adjective"""
        neighbors = self.knn.kneighbors(
            adj.embedding.numpy().reshape(1, -1), n_neighbors=k, return_distance=False
        )
        return list(map(lambda n: self.tensors[n], neighbors[0]))


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


def build_adjective_dict(adj2emb):
    """
    Build adjective dict using wordnet and the given
    adj2emb dictionary for adjective word embeddings.
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
                    embedding = torch.from_numpy(adj2emb[word_name])
                    word2adj[word_name] = Adjective(word_name, embedding, set(), set())
                except KeyError:
                    continue

            adj = word2adj[word_name]
            adj.antonyms = adj.antonyms | current_antonyms
            adj.hyponyms = adj.hyponyms | current_hyponyms

    return word2adj


def build_training_triples(model, filtered=None, restricted=False):
    """
    Builds a list of <adjective, cohyponym, antonym> triples
    for the given model. The model contains all the adjectives,
    and allows querying for embeddings using antonym names.

    Optionally takes an enumerable of filtered words from
    which we filter triples where the input adjective is in that
    enumerable.

    An additional option `restricted` represents whether hyponyms
    for a word in filtered should be filtered as well.
    """
    filtered = [] if filtered is None else filtered

    if restricted:
        for f in filtered:
            try:
                filter_adj = model.adj_from_name(f)
                filtered = filtered | filter_adj.hyponyms
            except KeyError:
                continue

    triples = []
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
                    triples.append((adj_emb, centroid, ant_emb))
                except KeyError:
                    # print("failed for %s and %s" % (adj.name, ant))
                    continue

    return triples


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


def load_adj2emb(path=ADJ2EBM_PATH):
    """
    Loads the <adj, emb> triples created with preprocess.py
    into a dictionary.
    """
    with open(path, "r") as f:
        adj2emb = {}
        for line in f:
            adj, emb_str = line.split("\t")
            # trim away brackets when loading
            emb = np.fromstring(emb_str[1:-1], sep=",")
            adj2emb[adj] = emb

        return adj2emb


def build_triples_and_adj_model(restricted=False):
    """
    Simply access function running the entire Adjective Model
    and training triple creation, including filtered based on GRE
    all in one step.

    Takes a parameter whether the filtering should be restricted
    or not, as defined in Rimell 2018.

    Returns a tuple with <train_triples, AdjectiveModel>.
    """
    adj2emb = load_adj2emb(ADJ2EBM_PATH)
    adj2adj = build_adjective_dict(adj2emb)
    adj_model = AdjectiveModel(adj2adj)
    filtered_words = load_gre_filtered_words()
    triples = build_training_triples(adj_model, filtered_words, restricted)
    return triples, adj_model


def main():
    """Build model and print length of training triples"""
    # Load the Google news pre-trained Word2Vec model
    triples, _ = build_triples_and_adj_model()
    print(len(triples))

if __name__ == "__main__":
    main()
