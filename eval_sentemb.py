import os
import argparse
import json
import torch
from scipy.stats import spearmanr
import pytablewriter as ptw
from sentence_transformers import SentenceTransformer

from utils import read_file, cosine_sim

NUM_FOLD = 10

class STEmbedder:
    """Sentence Transformer embedding"""

    def __init__(self, model_name_or_path, device='cuda', cache_folder=None):
        self.model = SentenceTransformer(model_name_or_path, cache_folder=cache_folder).to(device)
        self.device = device

    def compute_embeddings(self, text, batch_size=128, normalize=True):
        embeddings_np = self.model.encode(text,
                                          batch_size=batch_size,
                                          show_progress_bar=False,
                                          device=self.device,
                                          normalize_embeddings=normalize)
        return embeddings_np


def predict(s1_embeddings, s2_embeddings):
    scores = []
    for e1, e2 in zip(s1_embeddings, s2_embeddings):
        scores.append(cosine_sim(e1, e2))
    return scores


def main(args):
    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedder = STEmbedder(args.model_name_or_path, device=device, cache_folder=args.cache_dir)

    # evaluation
    all_results = {}
    for l in args.languages:
        print(f"Language: {l}")
        results = {}
        for k in range(NUM_FOLD):
            _, sentence_pairs, true_scores = read_file(f"{args.data_dir}/{l}/{l}_dev_{k}.csv")

            ################## predict the scores  ##################
            # compute embeddings
            s1 = [s[0] for s in sentence_pairs]
            s2 = [s[1] for s in sentence_pairs]
            s1_embeddings = embedder.compute_embeddings(s1)
            s2_embeddings = embedder.compute_embeddings(s2)
            pred_scores = predict(s1_embeddings, s2_embeddings)
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
    writer.value_matrix = [[f"Fold {k}"] + [round(all_results[l][str(k)], 2) for l in args.languages] for k in
                           range(NUM_FOLD)]

    writer.value_matrix.append(["avg"] + [round(all_results[l]['avg'], 2) for l in args.languages])
    writer.write_table()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--output_dir', type=str, default='./results/sentence_transformer')
    parser.add_argument('--language', type=str, default='all')
    parser.add_argument('--model_name_or_path', type=str,
                        default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    parser.add_argument('--cache_dir', type=str, default='')
    args = parser.parse_args()

    args.output_dir = f"{args.output_dir}/{args.model_name_or_path}"

    if args.language == 'all':
        args.languages = ['eng', 'amh', 'arq', 'ary', 'esp', 'hau', 'mar', 'tel']
    else:
        args.languages = [args.language]

    main(args)
