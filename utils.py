import pandas as pd
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from sentence_transformers import InputExample
import wandb
import lang2vec.lang2vec as l2v

CODE_MAPPING = {
    'eng' : 'en',
    'amh' : 'am',
    'arq' : 'ar',
    'ary' : 'ar',
    'esp' : 'es',
    'hau' : 'ha',
    'mar' : 'mr',
    'tel' : 'te'
}


def train_callback(loss, epoch, steps) :
    wandb.log({"train/loss" : loss,
               "train/epoch" : epoch,
               "train/global_step" : steps})


def eval_callback(score, epoch, steps, best_score) :
    wandb.log({"eval/Spearman" : score,
               "eval/epoch" : epoch,
               "eval/global_step" : steps,
               "eval/best_score" : best_score})


def read_file(file_path) :
    """Read the provided labeled data for semantic textual relatedness """
    df = pd.read_csv(file_path, header=0)
    ids = df['PairID'].values

    if 'Score' in df.columns :
        true_scores = df['Score'].values
    else :
        true_scores = [0.0] * len(ids)

    sentence_pairs = []
    for d in df['Text'].values :
        try :
            s1, s2 = d.split('\n')
        except :
            try :
                s1, s2 = d.split('\\n')  # hausa
            except :
                s1, s2 = d.split('\t')  # amh
        sentence_pairs.append([s1.strip(), s2.strip()])

    return ids, sentence_pairs, true_scores


def load_str_dataset(data_files) :
    dataset = DatasetDict()
    for split in data_files.keys() :
        file = data_files[split]
        if file is not None :
            ids, sentence_pairs, scores = read_file(file)
            sentence1 = [s1 for s1, s2 in sentence_pairs]
            sentence2 = [s2 for s1, s2 in sentence_pairs]

            dataset[split] = Dataset.from_dict({
                    'idx' : ids,
                    'sentence1' : sentence1,
                    'sentence2' : sentence2,
                    'label' : scores})

    return dataset


def load_stsb_dataset(data_files, language) :
    # support eng, esp
    if language in ['eng', 'esp'] :
        dataset = load_dataset("stsb_multi_mt", name=CODE_MAPPING[language])
        dataset['validation'] = dataset['dev']
        score_column = 'similarity_score'
    else :
        exist_data_files = {}
        for split in data_files.keys() :
            if data_files[split] is not None :
                exist_data_files[split] = data_files[split]
        dataset = load_dataset('csv', data_files=exist_data_files)
        score_column = 'score'

    def _add_label(example) :
        example['label'] = float(example[score_column] / 5)
        return example

    dataset = dataset.map(_add_label)
    return dataset


def create_str_dataset(file_path) :
    _, sentence_pairs, scores = read_file(file_path)
    examples = []
    for i, (s1, s2) in enumerate(sentence_pairs) :
        examples.append(
            InputExample(texts=[s1, s2], label=np.float32(scores[i]))
        )
    return examples


def create_stsb_dataset(data_file, language) :
    # support eng, esp
    if language in ['eng', 'esp'] :
        dataset = load_dataset("stsb_multi_mt", name=CODE_MAPPING[language], split='train')
        score_column = 'similarity_score'
    else :
        dataset = load_dataset('csv', data_files={'train' : data_file}, split='train')
        score_column = 'score'

    examples = []
    for i in range(len(dataset)) :
        row = dataset[i]
        examples.append(
            InputExample(texts=[row['sentence1'], row['sentence2']], label=float(row[score_column]) / 5)
        )
    return examples


def create_sick_dataset(language, split, num=-1) :
    assert language == 'eng'
    dataset = load_dataset('sick', split=split)
    score_column = 'relatedness_score'

    if num > 0 :  # subsample the dataset if required
        if 'idx' not in dataset.features.keys() :
            dataset = dataset.add_column('idx', column=list(range(len(dataset))))
            idx = np.random.choice(dataset['idx'], size=num, replace=False)
            dataset = dataset.select(idx)

    examples = []
    for i in range(len(dataset)) :
        row = dataset[i]
        examples.append(
            InputExample(texts=[row['sentence_A'], row['sentence_B']], label=float(row[score_column]) / 5)
        )
    return examples


def create_nli_dataset(language, num=-1) :
    # load data
    dataset = load_dataset('xnli', CODE_MAPPING[language], split='validation')  # 1500 samples

    if num > -1 :  # subsample the dataset if required
        if 'idx' not in dataset.features.keys() :
            dataset = dataset.add_column('idx', column=list(range(len(dataset))))
            idx = np.random.choice(dataset['idx'], size=num, replace=False)
            dataset = dataset.select(idx)

    # create contrastive learning set
    def add_to_samples(sent1, sent2, label) :
        if sent1 not in train_data :
            train_data[sent1] = {'contradiction' : set(), 'entailment' : set(), 'neutral' : set()}
        train_data[sent1][label].add(sent2)

    train_data = {}
    labels = ['entailment', 'neutral', 'contradiction']
    for i in range(len(dataset)) :
        row = dataset[i]
        sent1 = row['premise'].strip()
        sent2 = row['hypothesis'].strip()
        add_to_samples(sent1, sent2, labels[row['label']])
        add_to_samples(sent2, sent1, labels[row['label']])  # Also add the opposite

    # create training samples for sentence transformer
    train_samples = []
    for sent1, others in train_data.items() :
        if len(others['entailment']) > 0 and len(others['contradiction']) > 0 :
            train_samples.append(InputExample(
                texts=[sent1, np.random.choice(list(others['entailment'])),
                       np.random.choice(list(others['contradiction']))]))
            train_samples.append(InputExample(
                texts=[np.random.choice(list(others['entailment'])), sent1,
                       np.random.choice(list(others['contradiction']))]))

    return train_samples


def cosine_sim(u, v) :
    # cosine similarity
    score = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return score


def get_lang2vec(language_list):
    #language_list = ["eng", "amh", "mar", "ary", "tel","hau", "kin", "spa", "arq", "afr", "hin", "ind", "arb", "pan"]
    uriel_distances = l2v.distance(['syntactic', 'phonological', 'inventory'], *language_list)
    # use the average distance when using multiple vectors -> the smaller the distance, the more similar the languages
    avg_dis = np.mean(uriel_distances, axis=0)
    return avg_dis
