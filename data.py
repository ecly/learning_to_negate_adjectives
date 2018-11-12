"""
Logic for building the data using NLTK and the binary of Google's
300 dimensional word2vec embeddings, trained on Google News data.

The pre-trained embeddings can be downloaded here:
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
"""
import gensim
from nltk.corpus import wordnet as wn

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
                    embedding = model.get_vector(word_name)
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
    Builds a list of adjective/antonym embedding pairs
    for the given adj_dict and model. The model is used
    for looking up the embeddings from an antonym name.
    """
    pairs = []
    for adj in adj_dict.values():
        for ant in adj.antonyms:
            try:
                ant_emb = model.get_vector(ant)
                pairs.append((adj.embedding, ant_emb))
            except KeyError:
                continue

    return pairs


def main():
    # Load the Google news pre-trained Word2Vec model
    model = gensim.models.KeyedVectors.load_word2vec_format(
        GOOGLE_NEWS_PATH, binary=True
    )
    adj_dict = build_adjective_dict(model)
    print(sum(map(lambda x: len(x.antonyms), adj_dict.values())))
    pairs = build_training_pairs(adj_dict, model)
    print(len(pairs))


if __name__ == "__main__":
    main()
