# Learning to Negate Adjectives with Bilinear Models
Repository replicating the results from the paper [Learning to Negate Adjectives with Bilinear Models](https://aclweb.org/anthology/E17-201://aclweb.org/anthology/E17-2012) (2017) by Laura Rimell, Amandla Mabona, Luana Bulat and Douwe Kiela.

Thanks to Laura Rimell for supplying the test data for the original experiments.

A pre-trained standard model trained for 175 epochs (200 used in original paper) can be downloaded [here](https://drive.google.com/open?id=193yw5ch1ekp-1o8y75IS9La0ZM2oEm3V).

### Requirements
`pip3 install -r requirements.txt`  
`python3 -c "import nltk; nltk.download('wordnet')"`

### Implementation
Examples for running [preprocess.py](preprocess.py), [train.py](train.py), [evaluate.py](evaluate.py) and [test.py](test.py) are given at the top of the files respectively.

##### [preprocess.py](preprocess.py)
Produces [adjective_embeddings.txt](data/adjective_embeddings.tsv) using GoogleNews 300 dimensional word2vec embeddings.

##### [data.py](data.py)
Logic for parsing WordNet, test data and for building the various variations of the training data ['*standard*', '*restricted*', '*unsupervised*']. Also contains the `AdjectiveModel` which is the wrapping structure for keeping track of adjective's antonyms/cohyponyms as well as querying for k-nearest-neighbors.

##### [model.py](model.py)
Implementation of PyTorch modules for Encoder, Decoder and their wrapper EncoderDecoder.

##### [train.py](train.py)
Load/save/train models. Supports multi-GPU training using `DataLoaders` and `nn.DataParallel`. See `python train.py --help` for options.

##### [evaluate.py](evaluate.py)
Run and print evaluation results for Experiment 1 and 2 as described in Rimell et al. 2018. Uses test data present in the [data](data) directory.

##### [test.py](test.py)
Easy way to predict and print antonyms for adjectives either given as arguments or added to the python file itself. Antonyms printed for a word are marked with [] if they are present in the gold standard.
