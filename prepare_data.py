from sklearn.model_selection import KFold
import pandas as pd
import random
import os
import argparse

# NUM_FOLD=10
NUM_FOLD = 5
SEED = 42

def split_str_data(data_dir, output_dir, languages) :
    for language in languages :
        print(f"Processing {language}...")

        train_file = os.path.join(data_dir, f"{language}/{language}_train.csv")

        if os.path.isfile(train_file) :
            print('Split the original train.csv into 5 folds for cross-validation')
            output_dir = os.path.join(output_dir, f"{language}/5_fold/")
            os.makedirs(output_dir, exist_ok=True)

            label_data = pd.read_csv(train_file, header=0)

            # split the labeled training data into 5 folds
            kf = KFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED)
            for i, (train_index, test_index) in enumerate(kf.split(label_data)) :
                train_subset = label_data.iloc[train_index]
                dev_subset = label_data.iloc[test_index]
                train_subset.to_csv(os.path.join(output_dir, f"{language}_train_{i}.csv"), index=False)
                dev_subset.to_csv(os.path.join(output_dir, f"{language}_dev_{i}.csv"), index=False)
                print(f"Fold {i}:")
                print(f"Train size: {len(train_subset)}")
                print(f"Dev size: {len(dev_subset)}")


def create_tapt_data(data_dir, output_dir, languages) :
    for language in languages :
        print(f"Processing {language}...")

        # if pilot data exists, rename it
        train_file = os.path.join(data_dir, f"{language}/{language}_train.csv")
        dev_file = os.path.join(data_dir, f"{language}/{language}_dev.csv")
        test_file= os.path.join(data_dir, f"{language}/{language}_test.csv")

        unlabeled_text = []

        for file in [train_file, dev_file, test_file]:
            if os.path.isfile(file) :
                print(file)
                unlabeled_text.extend(pd.read_csv(file, header=0)['Text'].values)

        with open(os.path.join(output_dir, f"{language}.txt"), "w", encoding='utf8') as f :
            for row in unlabeled_text :
                #use special token to concatenate sentences
                f.write("</s></s>".join(row.split("\n")) + '\n')


def prepare_mono_data(download_dir, output_dir, languages) :
    for language in languages :
        print(f"Processing {language}...")
        cur_dir = os.path.join(download_dir, language)
        sentences = []

        files = os.listdir(cur_dir)
        files = [f for f in files if 'sentences.txt' in f]

        for file in files :
            filename = os.path.join(cur_dir, file)
            with open(filename, 'r') as f :
                rows = f.readlines()
                rows = [r.split('\t')[1].strip() for r in rows]
                sentences.extend(rows)

        random.seed(SEED)
        random.shuffle(sentences)

        with open(os.path.join(output_dir, f"{language}.txt"), "w", encoding='utf8') as f :
            for row in sentences :
                f.write(row + '\n')


def combine_data(data_dir, output_dir, source_lang, target_lang) :
    print(f"Combining {source_lang} and {target_lang}...")
    dir_1 = os.path.join(data_dir, source_lang)
    dir_2 = os.path.join(data_dir, target_lang)

    for fold in range(NUM_FOLD) :
        print(f"Fold {fold}...")
        train_file_1 = os.path.join(dir_1, f"5_fold/{source_lang}_train_{fold}.csv")
        train_file_2 = os.path.join(dir_2, f"5_fold/{target_lang}_train_{fold}.csv")

        train_1 = pd.read_csv(train_file_1, header=0)
        train_2 = pd.read_csv(train_file_2, header=0)

        train = pd.concat([train_1, train_2], axis=0)

        train.to_csv(os.path.join(output_dir, f"{target_lang}_{source_lang}_train_{fold}.csv"), index=False)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/track_a/')
    parser.add_argument('--output_dir', type=str, default='./data/track_a/n_folds/')
    parser.add_argument('--languages', type=str, default='eng', help='Languages to process, separated by comma')
    parser.add_argument('--source_lang', type=str)
    parser.add_argument('--target_lang', type=str)
    args = parser.parse_args()

    # track_a_languages = ['amh', 'arq', 'ary', 'eng', 'esp', 'hau', 'kin', 'mar', 'tel']
    # track_c_languages = ['*afr*', 'amh', '*arb*', 'arq', 'ary', 'eng', 'esp', 'hau', '*hin*', '*ind*', 'kin', '*pan*']

    languages = [l.strip() for l in args.languages.split(',')]

    ## (1) Split the STR task data to NUM_FOLDS for cross-validation
    data_dir = './data/track_a/'
    output_dir = './data/track_a'
    split_str_data(args.data_dir, args.output_dir, languages)

    ## (2) Get the unlabeled sentences from STR task data for task adaptive pretraining
    # args.data_dir = './data/track_c/' # or './data/track_c/'
    # args.output_dir = './data/tapt/'
    # os.makedirs(args.output_dir, exist_ok=True)
    # create_tapt_data(args.data_dir, args.output_dir, languages)

    ## (3) Prepare the monolingual data for training language adapters
    #data_dir = './data/monolingual/'
    #output_dir = './data/monolingual/'
    #prepare_mono_data(data_dir, output_dir, languages)

    ## (4) Combine the data from two languages for training cross-lingual adapters
    #data_dir = './data/track_a/'
    #output_dir = './data/multilingual/'
    #combine_data(args.data_dir, args.output_dir, args.source_lang, args.target_lang)