import os
import re
import json
from scipy.stats import spearmanr
import pytablewriter as ptw

from utils import read_file
import argparse
import pandas as pd

NUM_FOLD = 10


def dice_score(s1, s2):
    # provided by the organizer -> remove stop words and/or punctuation
    s1 = s1.lower()
    s1_split = re.findall(r"\w+|[^\w\s]", s1, re.UNICODE)

    s2 = s2.lower()
    s2_split = re.findall(r"\w+|[^\w\s]", s2, re.UNICODE)

    dice_coef = len(set(s1_split).intersection(set(s2_split))) / (len(set(s1_split)) + len(set(s2_split)))
    return round(dice_coef, 2)


def predict(sentence_pairs):
    scores = []
    for s1, s2 in sentence_pairs:
        scores.append(dice_score(s1, s2))
    return scores


def main_fold(args):
    all_results = {}
    for l in args.languages:
        print(f"Language: {l}")
        results = {}
        for k in range(NUM_FOLD):
            _, sentence_pairs, true_scores = read_file(f"{args.data_dir}/{l}/{l}_dev_{k}.csv")

            ################## predict the scores  ##################
            pred_scores = predict(sentence_pairs)
            ###################################### ##################

            spearman = spearmanr(true_scores, pred_scores)[0]
            results[f"{k}"] = spearman * 100
        results['avg'] = round(sum(results.values()) / len(results.values()), 2)

        all_results[l] = results

        # write to disk
        os.makedirs(f"{args.output_dir}/{l}", exist_ok=True)
        with open(f"{args.output_dir}/{l}/result.json", 'w') as fp:
            json.dump(results, fp, indent=2)

    # print
    writer = ptw.MarkdownTableWriter()
    writer.headers = [""] + args.languages
    writer.value_matrix = [[f"Fold {k}"] + [round(all_results[l][str(k)], 2) for l in args.languages] for k in range(NUM_FOLD)]

    writer.value_matrix.append(["avg"] + [round(all_results[l]['avg'],2) for l in args.languages] )
    writer.write_table()

def main(args):
    for l in args.languages:
        print(f"Language: {l}")
        ids, sentence_pairs, true_scores = read_file(f"{args.data_dir}/{l}/{l}_test_with_labels.csv")
        ################## predict the scores  ##################
        pred_scores = predict(sentence_pairs)
        ###################################### ##################

        spearman = spearmanr(true_scores, pred_scores)[0]

        print(spearman)

        data = {
            'PairID': ids,
            'Pred_Score': pred_scores
        }
        df = pd.DataFrame.from_dict(data)
        df.to_csv(os.path.join(args.result_dir, f'overlap_{l}.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/semrel')
    parser.add_argument('--result_dir', type=str, default='./results/',
                        help='The output directory where the evaluation results will be written.')
    parser.add_argument('--eval_lang', type=str, default='ind',
                        help='The language to evaluate. Used for evaluation scripts only')
    args = parser.parse_args()

    args.output_dir = args.result_dir

    if args.eval_lang == 'all':
        args.languages = ['eng', 'amh', 'arq', 'ary', 'esp', 'hau', 'mar', 'tel']
    else:
        args.languages = [args.eval_lang]

    main(args)
