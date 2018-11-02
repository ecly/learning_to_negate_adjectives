from gensim.models import Word2Vec
from nltk.corpus import wordnet as wn

# Load the Google news pre-trained Word2Vec model
news_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
wn_model = wn.all_synsets()
