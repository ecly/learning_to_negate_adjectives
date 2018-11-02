import gensim
from nltk.corpus import wordnet as wn

GOOGLE_NEWS_PATH = "./GoogleNews-vectors-negative300.bin"

class Adjective:
    def __init__(self, name, embedding, hyponyms, antonyms):
        self.name = name
        self.embedding = embedding
        self.antonyms = antonyms
        self.hyponyms = hyponyms

    def __str__(self):
        return "%s, %s, %s" % (self.name, self.antonyms, self.hyponyms)


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


def build_adjective_dict(model):
    """
    Build adjective dict using wordnet and the given
    model for adjective word embeddings
    """
    word2antonyms = {}
    hyponym_groups = build_hyponym_groups()

    current_hyponyms = set()
    for synset in wn.all_synsets(wn.ADJ):
        if synset.pos() == "a":
            current_hyponyms = hyponym_groups[synset.name()]

        for word in synset.lemmas():
            word_name = word.name()

            if word_name not in word2antonyms:
                embedding = []
                word2antonyms[word_name] = Adjective(word_name, embedding, set(), set())

            adj = word2antonyms[word_name]
            for antonym in word.antonyms():
                adj.antonyms.add(antonym.name())

            adj.hyponyms = adj.hyponyms | current_hyponyms

    return word2antonyms

def main():
    # Load the Google news pre-trained Word2Vec model
    # model = gensim.models.KeyedVectors.load_word2vec_format(GOOGLE_NEWS_PATH, binary=True)

    model = []
    adj_dict = build_adjective_dict(model)
    print(adj_dict["dry"].hyponyms)

if __name__ == "__main__":
    main()
