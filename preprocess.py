"""
Module for preprocessing WordNet/GoogleNews model.
Creates a TSV-file containing all the adjectives present
in WordNet alongside their embedding from the GoogleNews
word2vec model.
"""
import sys
import gensim
from nltk.corpus import wordnet as wn
import numpy as np

GOOGLE_NEWS_PATH = "./GoogleNews-vectors-negative300.bin"
DEFAULT_TSV_PATH = "adj_emb.tsv"


def build_adjective_pairs(model):
    """
    Build and return list of <adjective, embedding> tuples
    using WordNet and the given gensim model.
    """
    pairs = []
    seen = set()
    for synset in wn.all_synsets(wn.ADJ):
        for word in synset.lemmas():
            name = word.name()
            if name in seen:
                continue
            try:
                emb = model.get_vector(name)
                pairs.append((name, emb))
                seen.add(name)
            except KeyError:
                continue

    return pairs


def print_pairs_to_file(pairs, path):
    """
    Prints the given pairs to a file at path.
    Format: <adj_name>\t<embedding>
    """
    with open(path, "w") as f:
        for name, emb in pairs:
            emb_str = np.array2string(emb, separator=",", max_line_width=sys.maxsize)
            print("%s\t%s" % (name, emb_str), file=f)


def main():
    inp_path = GOOGLE_NEWS_PATH if len(sys.argv) < 2 else sys.argv[1]
    out_path = DEFAULT_TSV_PATH if len(sys.argv) < 3 else sys.argv[2]
    gensim_model = gensim.models.KeyedVectors.load_word2vec_format(
        inp_path, binary=True
    )
    pairs = build_adjective_pairs(gensim_model)
    print_pairs_to_file(pairs, out_path)


if __name__ == "__main__":
    main()
