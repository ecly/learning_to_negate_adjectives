"""
Small script for illustrating some test examples ran on a given model.

Optionally takes additional words as command line arguments.

Predictions marked with [] in the output denote that they were present
in the gold standard antonyms for the given word.
"""
import sys
import torch
import data
from evaluate import predict_antonym_emb
import train

TESTS = ["ornate", "ruthless"]

def main(model_path, tests):
    """Run and print tests for `TESTS`-list on given model"""
    adj_model = data.build_adj_model()
    gold_standard = data.load_gold_standard(adj_model)
    model, _optimizer = train.initialize_model(model_path)
    with torch.set_grad_enabled(False):
        for test in tests:
            print("Predictions for %s:" % test)
            ant_pred = predict_antonym_emb(model, adj_model, test)
            predictions = [a.name for a in adj_model.adjs_from_vector(ant_pred, count=5)]
            output = []
            for prediction in predictions:
                if prediction in gold_standard:
                    output.append("[%s]" % prediction)
                else:
                    output.append("%s" % prediction)
            print("\t" + ", ".join(output))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Expected model path as command line argument")
        sys.exit(1)

    main(sys.argv[1], TESTS + [test.lower() for test in sys.argv[2:]])
