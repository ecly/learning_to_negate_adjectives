"""
Module for building the datasets for training and evaluation
using WordNet and the <adjective, embedding> file output by
the preprocess module.

This also include logic for building the variations of the model in the
form of `restricted` and `unsupervised`. Both of these are present
as parameters to `build_dataset` and should be used from the train module.
"""
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from nltk.corpus import wordnet as wn
from sklearn.neighbors import NearestNeighbors
import numpy as np

CENTROID_MIN_BASIS = 10
ANTONYM_THESAURUS = "./data/thesaurus_antonyms_extended"
LB_INPUT_WORDS = "./data/lb.inputwords"
GRE_INPUT_WORDS = "./data/gre_test_adjs_inputwords.txt"
GRE_TEST_QUESTIONS = "./data/gre_testset_adjs.txt"
ADJ2EBM_PATH = "./data/adjective_embeddings.tsv"


class Adjective:
    """
    Class encapsulating an Adjective with its name,
    embedding, (co)hyponyms and antonyms as per WordNet.
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
    An AdjectiveModel encapsulating mappings from
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
        np_vectors = [t.cpu().numpy() for t in tensors]
        knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="auto").fit(
            np_vectors
        )
        return knn

    def adj_from_name(self, name):
        """Get an Adjective from its name"""
        return self.adj2adj[name]

    def adj_from_vector(self, vector):
        """
        Get a single Adjective from a pytorch tensor embedding
        This is done by finding the nearest known adjective embedding
        to the given tensor and looking up its corresponding Adjective.

        Returns a one best Adjective instance.
        """
        neighbors = self.knn.kneighbors(
            vector.cpu().numpy().reshape(1, -1), n_neighbors=1, return_distance=False
        )
        return self.emb2adj[self.tensors[neighbors[0][0]]]

    def adjs_from_vector(self, vector, count=5):
        """
        Get `count` adjectives from a pytorch tensor embedding.
        This is represented as the `count` nearest known adjectives for the given
        vector, represented as Adjective instances.

        Returns a list of Adjective instances.
        """
        neighbors = self.knn.kneighbors(
            vector.cpu().numpy().reshape(1, -1), n_neighbors=count, return_distance=False
        )
        return list(map(lambda emb: self.emb2adj[emb], map(lambda nn: self.tensors[nn], neighbors[0])))

    def has_adj(self, name):
        """Returns True if given adj is known to model otherwise False"""
        return name in self.adj2adj

    def word_from_vector(self, vector):
        """Get name of the Adjective with embedding closest to given vector"""
        return self.adj_from_vector(vector).name

    def kneighbors(self, adj, k):
        """Get the tensors of the k-nearest neighbors to given Adjective"""
        neighbors = self.knn.kneighbors(
            adj.embedding.cpu().numpy().reshape(1, -1), n_neighbors=k, return_distance=False
        )
        return list(map(lambda n: self.tensors[n], neighbors[0]))


class AdjectiveDataset(Dataset):
    """General purpose pytorch Dataset"""

    def __init__(self, data):
        self.data = data
        self.len = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def calc_centroid(matrix):
    """
    Calculate centroid of list of torch tensors.

    Returns: 1D torch tensor.
    """
    return torch.mean(matrix, dim=0)


def find_gate_vector(adj, model, unsupervised=False):
    """
    Finds a gate vector for an adjective using the given model.

    If unsupervised=True is given, we ignore hyponyms from WordNet
    and instead only use nearest neighbors to create the gate vector.

    Otherwise we prioritize these over nearest neighbors from vector space.
    """
    hyp_count = 0 if unsupervised else len(adj.hyponyms)
    hyp_emb = [] if unsupervised else list(
        map(
            lambda a: model.adj_from_name(a).embedding,
            filter(model.has_adj, adj.hyponyms),
        )
    )
    filter_hyp_emb = set(hyp_emb) | {adj.embedding}

    if hyp_count < CENTROID_MIN_BASIS:
        neighbors = model.kneighbors(adj, CENTROID_MIN_BASIS)
        relevant = [n for n in neighbors if n not in filter_hyp_emb]
        missing_hyp = CENTROID_MIN_BASIS - hyp_count
        hyp_emb = hyp_emb + relevant[:missing_hyp]

    return calc_centroid(torch.stack(hyp_emb))


def build_hyponym_groups():
    """
    Hyponym groups are recognized by their 'lead' adjective.
    The one with synset.pos() = 'a'. All adjectives below an 'a'
    belong to that 'a's synset until another 'a' is met.
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
    Build adjective dict using WordNet and the given
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
            # skip words we don't have embeddings for
            if word_name not in adj2emb:
                continue

            if word_name not in word2adj and word_name in adj2emb:
                embedding = torch.from_numpy(adj2emb[word_name])
                word2adj[word_name] = Adjective(word_name, embedding, set(), set())

            adj = word2adj[word_name]
            adj.antonyms = adj.antonyms | current_antonyms
            adj.hyponyms = adj.hyponyms | current_hyponyms

    return word2adj

def load_gre_test_set(adj_model):
    """
    Loads and creates a test set of tuples
    <input, [options], answer> for antonym prediction.

    The given adj_model is used to remove questions where
    """
    with open(GRE_TEST_QUESTIONS, "r") as f:
        test_data = []
        for line in f:
            adj, rest = line.strip().lower().split(": ", 1)
            opt_str, answer = rest.split(" :: ")
            options = opt_str.split()
            # only keep questions where we know all words
            if all(adj_model.has_adj(x) for x in options + [adj, answer]):
                test_data.append((adj, options, answer))

        return test_data

def load_gre_words():
    """Loads and creates a set of the input words for the GRE test set """
    with open(GRE_INPUT_WORDS, "r") as f:
        return set(map(lambda w: w.strip().lower(), f))


def load_lb_words(path=LB_INPUT_WORDS):
    """Returns a list of the LB input words as strings"""
    with open(path, "r") as f:
        return list(map(lambda w: w.strip().lower(), f))

def load_gold_standard(adj_model):
    """
    Loads and creates a gold standard by combining the known antonyms
    based on WordNet, GRE and adjectives from Roget's 21st Century Thesaurus.
    Returns a dictionary from adjectives to a set of known antonyms.
    """
    data = defaultdict(set)
    with open(ANTONYM_THESAURUS, "r") as f:
        for line in f:
            adj, ant = line.strip().lower().split(" ", 1)
            data[adj].add(ant)

    gre_data = load_gre_test_set(adj_model)
    for adj, _, answer in gre_data:
        data[adj].add(answer)

    for adj, ants in data.items():
        if adj_model.has_adj(adj):
            model_ants = adj_model.adj_from_name(adj).antonyms
            ants.update(model_ants)

    return data


def load_adj2emb(path=ADJ2EBM_PATH):
    """
    Loads the <adj, emb> pairs created with 'preprocess.py' into a dictionary.
    """
    with open(path, "r") as f:
        adj2emb = {}
        for line in f:
            adj, emb_str = line.split("\t")
            # trim away brackets when loading
            emb = np.fromstring(emb_str[1:-1], sep=",")
            adj2emb[adj] = emb

        return adj2emb


def build_filtered_words(adj_model):
    """
    Builds a set of filtered words using the Gold Standard
    which encapsulates both input words from GRE and LB.
    """
    return set(load_gold_standard(adj_model).keys())


def build_adj_model():
    """
    Builds the AdjectiveModel using embeddings at `ADJ2EMB_PATH`
    as well as adjective/antonyms/hyponyms found from WordNet.
    """
    adj2emb = load_adj2emb(ADJ2EBM_PATH)
    adj2adj = build_adjective_dict(adj2emb)
    return AdjectiveModel(adj2adj)


def build_dataset(adj_model=None, custom_filter=None, restricted=False, unsupervised=False):
    """
    Builds an AdjectiveDataset for the given model.
    The model contains all the adjectives, and allows querying
    for embeddings using antonym names.

    Optionally takes a custom enumerable of filtered words from
    which we filter triples where the input adjective is in that
    enumerable. If no custom_filter is used, the default combination
    of words from GRE questions and Gold standard input words are used.

    Option `restricted` represents whether hyponyms for a word in filtered
    should be filtered as well.

    Option `unsupervised` indicates whether WordNet hyponyms should be included
    when calculating the gate vector.
    """
    adj_model = build_adj_model() if adj_model is None else adj_model
    filtered = (
        build_filtered_words(adj_model) if custom_filter is None else custom_filter
    )

    if restricted:
        for f in filtered:
            if adj_model.has_adj(f):
                filter_adj = adj_model.adj_from_name(f)
                filtered = filtered | filter_adj.hyponyms

    triples = []
    for adj in adj_model.adj2adj.values():
        adj_name = adj.name
        if adj.name in filtered:
            continue

        current_adj = adj_model.adj_from_name(adj_name)
        adj_emb = current_adj.embedding
        centroid = find_gate_vector(current_adj, adj_model, unsupervised)

        for ant_name in filter(adj_model.has_adj, adj.antonyms):
            ant_emb = adj_model.adj_from_name(ant_name).embedding
            triples.append((adj_emb, centroid, ant_emb))

    return AdjectiveDataset(triples)


def main():
    """Build model and print length of size of training set"""
    dataset = build_dataset()
    print(len(dataset.data))


if __name__ == "__main__":
    main()
