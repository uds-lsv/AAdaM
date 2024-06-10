import os
import torch
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sentence_transformers import CrossEncoder

from utils import read_file


class CrossEncoderEvaluator :
    def __init__(self, model_name_or_path, batch_size=128, device='cuda') :
        self.model = CrossEncoder(model_name_or_path, num_labels=1, device=device)
        self.device = device
        self.batch_size = batch_size

    def compute_scores(self, sentence_pairs) :
        pred_scores = self.model.predict(sentence_pairs,
                                         batch_size=self.batch_size,
                                         convert_to_numpy=True,
                                         show_progress_bar=False)
        return pred_scores


def eval(model_list, data_file, output_file, compute_metrics) :
    metrics = []
    pred_scores = []
    for model in model_list :
        print(f"Evaluating {model} ... ")
        # load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = CrossEncoderEvaluator(model, device=device)

        ids, sentence_pairs, labels = read_file(data_file)

        ################## predict the scores  ##################
        predictions = model.compute_scores(sentence_pairs)
        ###################################### ##################
        pred_scores.append(predictions)

        if args.compute_metrics :
            metric = spearmanr(predictions, labels)[0] * 100
            metrics.append(metric)

    ensemble_scores = np.mean(np.array(pred_scores), axis=0)

    data = {
        'PairID' : ids,
        'Pred_Score' : ensemble_scores
    }
    df = pd.DataFrame.from_dict(data)
    df.to_csv(output_file, index=False)

    log_scores = np.transpose(np.array(pred_scores))
    log_scores = np.array([str(l).strip("[]") for l in log_scores])
    log_data = {
        'PairID': ids,
        'Pred_Score': ensemble_scores,
        'Log_Scores': log_scores
    }

    log_df = pd.DataFrame.from_dict(log_data)
    log_df.to_csv(output_file + '.log', index=False)

    if compute_metrics :
        print(f"spearmanr: {metrics}")
        print(f"Avg/std spearmanr: {round(np.mean(metrics), 2)} ± {round(np.std(metrics), 2)}")
        with open(output_file + '.metric', 'a') as f :
            f.write(f"model: {model_list}\n")
            f.write(f"spearmanr: {metrics}\n")
            f.write(f"Avg/std spearmanr: {round(np.mean(metrics), 2)} ± {round(np.std(metrics), 2)}\n")


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_list', type=str, default=None,
                        help='The model(s) to evaluate separated by comma. We will ensemble their results.')
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--eval_lang', type=str, default=None, help='The language to evaluate. ')
    parser.add_argument('--track', type=str, choices=['a', 'c'], default='a')
    parser.add_argument('--output_dir', type=str, default='./submission/',
                        help='The output directory where the score file will be written.')
    parser.add_argument('--compute_metrics', action='store_true')
    args = parser.parse_args()

    print(args)

    os.makedirs(args.output_dir, exist_ok=True)

    model_list = args.model_list.split(',')
    output_file = os.path.join(args.output_dir, f"pred_{args.eval_lang}_{args.track}.csv")

    eval(model_list, args.data_file, output_file, args.compute_metrics)
