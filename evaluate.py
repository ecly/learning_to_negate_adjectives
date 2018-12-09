"""
Evaluation code for both Gold Standard and GRE Questions
Examples:
    python evaluate.py
    python evaluate.py adjective_negation_model.tar
    python evaluate.py adjective_negation_model.tar cpu
    python evaluate.py adjective_negation_model.tar cuda
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import data
import train

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_cosine(tensor1, tensor2, device=DEVICE):
    """
    Compute cosine similarity between two pytorch tensors
    Returns a regular python float where 1.0 means identical
    tensors and 0.0 means orthoganal tensors.
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
    Evaluate the given model according to GRE
    question set. The given adj_model is needed to compute
    embeddings for the GRE adjectives.

    Optionally takes a loaded GRE dataset to avoid loading
    multiple times, if evaluation is ran repeatedly.
    """
    with torch.set_grad_enabled(False):
        gre_data = data.load_gre_test_set(adj_model) if gre is None else gre
        right = []
        wrong = []
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


def evaluate_gold_standard(model, adj_model, device=DEVICE, gold=None):
    """
    Evaluate the given model according to the gold standard.
    The given adj_model is needed to compute embeddings for
    the gold standard adjectives.
    """
    with torch.set_grad_enabled(False):
        gold_data = data.load_gold_standard(adj_model) if gold is None else gold
        right = []
        wrong = []
        for adj_str, antonyms in gold_data.items():
            ant_pred = predict_antonym_emb(model, adj_model, adj_str, device)
            ant_name = adj_model.adj_from_vector(ant_pred).name
            if ant_name in antonyms:
                # print("Correct: %s %s" %(adj_str, ant_name))
                right.append((adj_str, ant_name))
            else:
                # print("Wrong: %s %s" %(adj_str, ant_name))
                wrong.append((adj_str, ant_name))

        return right, wrong


def evaluate(model, adj_model, device=DEVICE):
    """
    Evaluates the given model on both GRE 5 option questions
    and on the gold standard. Results are printed to stdout.
    """
    gre_data = data.load_gre_test_set(adj_model)
    gre_right, _gre_wrong = evaluate_gre(model, adj_model, device, gre_data)
    gre_acc = len(gre_right) / len(gre_data)
    print("GRE question accuracy: %.2f" % gre_acc)

    gold_data = data.load_gold_standard(adj_model)
    gold_right, _gold_wrong = evaluate_gold_standard(
        model, adj_model, device, gold_data
    )
    gold_acc = len(gold_right) / len(gold_data)
    print("Gold standard accuracy: %.2f" % gold_acc)


def main(model_path, device=DEVICE):
    """
    Loads the adj_model and EncoderDecoder model from a checkpoint.
    The EncoderDecoder model is then evaluated on both GRE question
    answer set and on the gold standard antonym prediction task.
    """
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
        device_type = sys.argv[2]
        assert device_type in ["cpu", "cuda"]
        main(sys.argv[1], torch.device(device_type))
