import os
import json
import fasttext
import numpy as np
import nltk
import argparse
from scipy.stats import spearmanr
from nltk.tokenize import word_tokenize
import pytablewriter as ptw

from utils import read_file, cosine_sim

nltk.download('punkt')

NUM_FOLD = 10

EMBEDDING = {
    'eng': 'wiki.en',
    'amh': 'wiki.am',
    'arq': 'wiki.ar',
    'ary': 'wiki.ar',
    'esp': 'wiki.es',
    'hau': 'wiki.ha',
    'mar': 'wiki.mr',
    'tel': 'wiki.te'
}

def get_embedding(sentence, emb_model):
    toks = word_tokenize(sentence.lower())
    word_vecs = [emb_model.get_word_vector(t) for t in toks]
    avg_vec = np.mean(np.array(word_vecs), axis=0)
    return avg_vec


def predict(sentence_pairs, emb_model):
    scores = []
    for s1, s2 in sentence_pairs:
        emb1 = get_embedding(s1, emb_model)
        emb2 = get_embedding(s2, emb_model)
        scores.append(cosine_sim(emb1, emb2))
    return scores


def main(args):
    all_results = {}
    for l in args.languages:
        print(f"Language: {l}")

        # load fasttext
        emb_model = fasttext.load_model(os.path.join(args.emb_dir, f'{EMBEDDING[l]}.bin'))

        results = {}
        for k in range(NUM_FOLD):
            _, sentence_pairs, true_scores = read_file(f"{args.data_dir}/{l}/{l}_dev_{k}.csv")

            ################## predict the scores  ##################
            pred_scores = predict(sentence_pairs, emb_model)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--result_dir', type=str, default='./results/',
                        help='The output directory where the evaluation results will be written.')
    parser.add_argument('--emb_dir', type=str, default='./fasttext/',
                        help='The directory where the static embeddings are stored.')
    parser.add_argument('--cache_dir', type=str, default='')
    parser.add_argument('--eval_lang', type=str, default='all',
                        help='The language to evaluate. Used for evaluation scripts only')
    parser.add_argument('--eval_batch_size', type=int, default=256)
    args = parser.parse_args()

    args.output_dir = args.result_dir

    if args.eval_lang == 'all':
        args.languages = ['eng', 'amh', 'arq', 'ary', 'esp', 'hau', 'mar', 'tel']
    else:
        args.languages = [args.eval_lang]

    main(args)
