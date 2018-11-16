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

CENTROID_MIN_BASIS = 10
GOOGLE_NEWS_PATH = "./GoogleNews-vectors-negative300.bin"


def word_from_vector(vector, model):
    return model.most_similar(positive=[vector.numpy()], topn=1)[0][0]


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


def calc_centroid(matrix):
    """
    Calculate centroid of list of torch tensors.

    Returns: 1D torch tensor.
    """

    return torch.mean(matrix, dim=0)


def find_gate_vector(adjective, model):
    """
    TESTME
    """
    hyponym_count = len(adjective.hyponyms)
    hyponyms = list(map(model.get_vector, adjective.hyponyms))
    filter_hyponyms = adjective.hyponyms | {adjective.name}

    if hyponym_count < CENTROID_MIN_BASIS:
        neighbors = model.similar_by_word(adjective.name, topn=hyponym_count + CENTROID_MIN_BASIS)
        relevant = list(map(lambda x: torch.from_numpy(model.get_vector(x[0])),
                            filter(lambda x: x[0] not in filter_hyponyms,
                                   neighbors)))
        # print(list(map(lambda x: word_from_vector(x, model), relevant)))
        missing_hyponyms = CENTROID_MIN_BASIS - hyponym_count
        hyponyms = hyponyms + relevant[:missing_hyponyms]

    return calc_centroid(torch.tensor(hyponyms))


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
                    word2adj[word_name] = Adjective(
                        word_name, embedding, set(), set()
                    )
                except KeyError:
                    continue

            adj = word2adj[word_name]
            adj.antonyms = adj.antonyms | current_antonyms
            adj.hyponyms = adj.hyponyms | current_hyponyms

    return word2adj


def build_training_pairs(adj_dict, model):
    """
    Builds a list of <adjective, cohyponym, antonym> triples
    for the given adj_dict and model. The model is used
    for looking up the embeddings from an antonym name.
    """
    pairs = []
    for adj in adj_dict.values():
        for ant in adj.antonyms:
            # print(adj.name, ant)
            try:
                ant_emb = torch.from_numpy(model.get_vector(ant))
                centroid = find_gate_vector(adj, model)
                pairs.append((adj.embedding, centroid, ant_emb))
            except KeyError:
                print("failed for %s and %s" % (adj.name, ant))
                continue

    return pairs

def main():
    # Load the Google news pre-trained Word2Vec model
    model = gensim.models.KeyedVectors.load_word2vec_format(
        GOOGLE_NEWS_PATH, binary=True)
    adj_dict = build_adjective_dict(model)
    # print(sum(map(lambda x: len(x.antonyms), adj_dict.values())))
    pairs = build_training_pairs(adj_dict, model)
    readable_pairs = list(map(lambda x: (word_from_vector(x[0], model), word_from_vector(x[2], model)), pairs))
    print(readable_pairs)
    # print(len(pairs))


if __name__ == "__main__":
    main()
