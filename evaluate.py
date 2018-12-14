"""
Evaluation code for both Gold Standard and GRE Questions
Examples:
    python evaluate.py adjective_negation_model.tar
    python evaluate.py adjective_negation_model.tar cpu
    python evaluate.py adjective_negation_model.tar cuda
"""
import sys
import torch
import torch.nn.functional as F
import data
import train

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In the original paper they say that several of the antonym
# predictions for P5 should be in the gold standard. According
# to Merriam Webster, several is more 2 and less than many, so
# we will use 3 or more correct predictions as a correct prediction.
SEVERAL = 3


def compute_cosine(tensor1, tensor2, device=DEVICE):
    """
    Compute cosine similarity between two pytorch tensors
    Returns a regular python float where 1.0 means identical
    tensors and 0.0 means orthogonal tensors.
    """
    tensor1 = tensor1.to(device)
    tensor2 = tensor2.to(device)
    return F.cosine_similarity(tensor1, tensor2, dim=0).item()


def predict_antonym_emb(model, adj_model, adj_str, device=DEVICE):
    """
    Predicts an antonym embedding using the given model, and adj_model
    to lookup the embedding for the given adj_str.
    """
    adj = adj_model.adj_from_name(adj_str)
    gate = data.find_gate_vector(adj, adj_model)

    x, z = adj.embedding.to(device), gate.to(device)
    y_pred = model(x, z)
    return y_pred


def evaluate_gre(model, adj_model, device=DEVICE, gre=None):
    """
    Evaluate the given model according to GRE question set.
    The given adj_model is needed to compute embeddings for the GRE adjectives.

    Optionally takes a loaded GRE dataset to avoid loading
    multiple times, if evaluation is ran repeatedly.
    """
    with torch.set_grad_enabled(False):
        gre_data = data.load_gre_test_set(adj_model) if gre is None else gre
        right, wrong = [], []
        for test in gre_data:
            adj_str, options, answer = test
            ant_pred = predict_antonym_emb(model, adj_model, adj_str, device)

            most_similar = 0
            most_similar_word = ""
            for opt_str in options:
                opt = adj_model.adj_from_name(opt_str)
                similarity = compute_cosine(ant_pred, opt.embedding)
                if similarity > most_similar:
                    most_similar = similarity
                    most_similar_word = opt_str

            if most_similar_word == answer:
                right.append(test)
            else:
                wrong.append(test)

        return right, wrong


def evaluate_gold_standard_p1(input_words, model, adj_model, device=DEVICE, gold=None):
    """
    Evaluate the given model and input_words according to the gold standard.
    `input_words` is a list of strings, which are adjectives that will
    have their antonym predicted to be evaluated against the gold standard.

    Here we predict a one best antonym and checks whether it is present in
    the gold standard. If an input word is given that we don't have embeddings
    for, it is skipped.

    The given adj_model is needed to compute embeddings for the input_words.
    The model is used for prediction. Optionally takes the gold standard as
    an input to avoid loading it multiple times under some circumstances.

    Returns two lists of (right, wrong) predictions respectively.
    """
    with torch.set_grad_enabled(False):
        gold = data.load_gold_standard(adj_model) if gold is None else gold
        right, wrong = [], []
        for adj_str in filter(adj_model.has_adj, input_words):
            antonyms = gold[adj_str]
            ant_pred = predict_antonym_emb(model, adj_model, adj_str, device)
            ant_name = adj_model.adj_from_vector(ant_pred).name
            if ant_name in antonyms:
                right.append((adj_str, ant_name))
            else:
                wrong.append((adj_str, ant_name))

        return right, wrong


def evaluate_gold_standard_p5(input_words, model, adj_model, device=DEVICE, gold=None):
    """
    Evaluate the given model and input_words according to the gold standard.
    input_words is a list of strings, which are the ones that will
    have their antonym predicted to be evaluated against the gold standard.

    The given adj_model is needed to compute embeddings for the input_words.
    The model is used for prediction. Optionally takes the gold standard as
    an input to avoid loading it multiple times under some circumstances.

    Returns two lists of (right, wrong) predictions respectively.
    """
    with torch.set_grad_enabled(False):
        gold = data.load_gold_standard(adj_model) if gold is None else gold
        right, wrong = [], []
        for adj_str in filter(adj_model.has_adj, input_words):
            antonyms = gold[adj_str]
            ant_pred = predict_antonym_emb(model, adj_model, adj_str, device)
            p5_antonyms = {
                a.name for a in adj_model.adjs_from_vector(ant_pred, count=5)
            }
            good, bad = [], []
            for ant in p5_antonyms:
                if ant in antonyms:
                    good.append(ant)
                else:
                    bad.append(ant)

            result_pair = (adj_str, good, bad)
            if len(good) >= SEVERAL:
                right.append(result_pair)
            else:
                wrong.append(result_pair)

        return right, wrong


def evaluate(model, adj_model, device=DEVICE):
    """
    Evaluates the given model on both GRE 5 option questions
    and GRE/LB P1/P5 against the Gold Standard. Results are printed to stdout.
    """
    # Experiment 1: GRE Question, 5 Options
    gre_data = data.load_gre_test_set(adj_model)
    gre_right, gre_wrong = evaluate_gre(model, adj_model, device, gre_data)
    gre_acc = len(gre_right) / (len(gre_right) + len(gre_wrong))
    print("GRE question accuracy: %.2f" % gre_acc)

    # Experiment 2: Gre P1/P5 Gold standard
    gold_data = data.load_gold_standard(adj_model)
    gre_words = data.load_gre_words()
    gre_right_p1, gre_wrong_p1 = evaluate_gold_standard_p1(
        gre_words, model, adj_model, device, gold_data
    )
    gre_acc_p1 = len(gre_right_p1) / (len(gre_right_p1) + len(gre_wrong_p1))
    print("GRE P1 accuracy: %.2f" % gre_acc_p1)

    gre_right_p5, gre_wrong_p5 = evaluate_gold_standard_p5(
        gre_words, model, adj_model, device, gold_data
    )
    gre_acc_p5 = len(gre_right_p5) / (len(gre_right_p5) + len(gre_wrong_p5))
    print("GRE P5 accuracy: %.2f" % gre_acc_p5)

    # Experiment 2: LB P1/P5 Gold standard
    lb_words = data.load_lb_words()
    lb_right_p1, lb_wrong_p1 = evaluate_gold_standard_p1(
        lb_words, model, adj_model, device, gold_data
    )
    lb_acc_p1 = len(lb_right_p1) / (len(lb_right_p1) + len(lb_wrong_p1))
    print("LB P1 accuracy: %.2f" % lb_acc_p1)

    lb_right_p5, lb_wrong_p5 = evaluate_gold_standard_p5(
        lb_words, model, adj_model, device, gold_data
    )
    lb_acc_p5 = len(lb_right_p5) / (len(lb_right_p5) + len(lb_wrong_p5))
    print("LB P5 accuracy: %.2f" % lb_acc_p5)


def main(model_path, device=DEVICE):
    """
    Loads the adj_model and EncoderDecoder model from a checkpoint.
    The EncoderDecoder model is then evaluated on both GRE question
    answer set and on GRE/LB P1/P5 gold standard antonym prediction task.
    """
    print("Building dataset and adjectives")
    adj_model = data.build_adj_model()
    model, _optimizer = train.initialize_model(model_path, device)
    model.eval()
    evaluate(model, adj_model, device)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Expected path for model as argument")
        sys.exit(1)
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        device_type = sys.argv[2].lower()
        assert device_type in ["cpu", "cuda"]
        main(sys.argv[1], torch.device(device_type))
