import argparse
import glob
import os
import random
import re

import pandas as pd

ESSAY_COL, CORRECTED_COL = 'essay', 'corrected'
INPUT_COL, OUTPUT_COL = 'input_text', 'target_text'
COMBINE_WHITESPACE = re.compile(r"\s+")
AUGMENTATION_DIR = {
    ' un ': ' una ',
    ' una ': ' un ',
    ' la ': ' el ',
    ' el ': ' la ',
    ' lo ': ' la '
}

TRAIN_PERC, VAL_PREC = 0.75, 0.15


def augment_data(train_df, num_samples=1000):
    """Augment input data by replacing tokens with common mistakes"""
    augmented_samples = []
    for _, row in train_df.iterrows():
        for k, v in AUGMENTATION_DIR.items():
            if k in row[OUTPUT_COL]:
                augmented_samples.append((row[OUTPUT_COL].replace(k, v), row[OUTPUT_COL]))

    samples = random.sample(augmented_samples, num_samples)
    return train_df.append(pd.DataFrame(samples, columns=[INPUT_COL, OUTPUT_COL]))


def preprocess_essay(essay):
    """Remove extra whitespace and quotations from the essay"""
    essay = essay.replace('"', "'")
    return COMBINE_WHITESPACE.sub(" ", essay).strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='Path to the `csv` folder in the COWS-L2H dataset')
    args = parser.parse_args()

    # Read all essays from all topics and quarters
    all_files = glob.glob(args.dataset_path + "/*.csv")
    dfs = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        dfs.append(df)

    # Clean empty rows and keep only original and corrected essay columns
    data = pd.concat(dfs, axis=0, ignore_index=True)
    corrected_data = data.dropna(subset=[CORRECTED_COL])
    sentence_pairs = corrected_data[[ESSAY_COL, CORRECTED_COL]]
    clean_sentence_pairs = sentence_pairs.applymap(preprocess_essay)

    print('Number of sentences in original and corrected essays respectively: {}, {}'.format(
        len([sentence for essay in clean_sentence_pairs.essay for sentence in essay.split('.')]),
        len([sentence for essay in clean_sentence_pairs.corrected for sentence in essay.split('.')])
    ))

    # Split all essays into sentences on period ('. ') boundary
    # Keep only those where the number of sentences in the original and the corrected essay are the same
    all_samples = []
    for idx, row in clean_sentence_pairs.iterrows():
        essay_sents = [sent.strip() for sent in row[ESSAY_COL].split('. ') if sent.strip()]
        corr_sents = [sent.strip() for sent in row[CORRECTED_COL].split('. ') if sent.strip()]

        if len(essay_sents) == len(corr_sents):
            all_samples.extend(list(zip(essay_sents, corr_sents)))

    random.Random(42).shuffle(all_samples)
    print('Total samples: {}; % corrected samples: {:.2}%'.format(len(all_samples), sum(
        [orig != corr for orig, corr in all_samples]) / len(all_samples)))

    # Split into train, validation, and test datasets
    train_boundary = int(len(all_samples) * TRAIN_PERC)
    val_boundary = train_boundary + int(len(all_samples) * VAL_PREC)

    train_df = pd.DataFrame(all_samples[:train_boundary], columns=[INPUT_COL, OUTPUT_COL])
    train_df_aug = augment_data(train_df, 1000)
    val_df = pd.DataFrame(all_samples[train_boundary:val_boundary], columns=[INPUT_COL, OUTPUT_COL])
    test_df = pd.DataFrame(all_samples[val_boundary:], columns=[INPUT_COL, OUTPUT_COL])

    # Write datasets to output tsvs
    data_dir = args.dataset_path.replace('csv', 'data')
    if not os.path.exists(data_dir): os.mkdir(data_dir)

    train_df.to_csv(data_dir + '/train.tsv', sep='\t', index=False)
    train_df_aug.to_csv(data_dir + '/train_aug.tsv', sep='\t', index=False)
    val_df.to_csv(data_dir + '/val.tsv', sep='\t', index=False)
    test_df.to_csv(data_dir + '/test.tsv', sep='\t', index=False)
