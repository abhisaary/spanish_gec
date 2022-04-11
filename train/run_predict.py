import argparse
import os

from torch.utils.data import DataLoader
from transformers import MT5Tokenizer

from train.cowsl2h import COWSL2H
from train.globals import DATASET_ARGS
from train.mt5_finetuner import MT5Finetuner


def clean_pred(pred):
    return pred.replace('<pad>', '').strip('<\s>').strip()


def predict_batch(model, batch, num_beams=2):
    """Uses the input model to generate a sequence of tokens until <\s> is output"""
    outs = model.model.generate(
        batch["source_ids"].cuda(),
        attention_mask=batch["source_mask"].cuda(),
        use_cache=True,
        decoder_attention_mask=batch['target_mask'].cuda(),
        max_length=DATASET_ARGS.max_seq_length,
        num_beams=num_beams,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    preds = [clean_pred(tokenizer.decode(ids)) for ids in outs]
    texts = [clean_pred(tokenizer.decode(ids)) for ids in batch['source_ids']]
    targets = [clean_pred(tokenizer.decode(ids)) for ids in batch['target_ids']]

    return preds, texts, targets


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_dir', type=str)
    parser.add_argument('ckpt_file', type=str)
    parser.add_argument('-d', '--datasets', type=str, choices=['train', 'val', 'test', 'all'], default='all')
    parser.add_argument('-b', '--num_beams', type=int, default=2)
    args = parser.parse_args()

    # Load the finetuned mT5 model from disc
    tokenizer = MT5Tokenizer.from_pretrained(DATASET_ARGS.tokenizer_name_or_path)
    model = MT5Finetuner.load_from_checkpoint(args.ckpt_file, params=DATASET_ARGS)
    model.to('cuda')
    output_dir = args.ckpt_dir + 'sentences/'
    os.mkdir(output_dir)

    # Output predictions for the train, val, and test sets
    datasets = [args.dataset] if args.dataset != 'all' else ['train', 'val', 'test']
    for data_split in datasets:
        print('Processing {} set'.format(data_split))
        with open(output_dir + '_'.join([data_split, 'combined.txt']), 'w') as f_comb, \
                open(output_dir + '_'.join([data_split, 'original.txt']), 'w') as f_orig, \
                open(output_dir + '_'.join([data_split, 'target.txt']), 'w') as f_tar, \
                open(output_dir + '_'.join([data_split, 'predicted.txt']), 'w') as f_pred:

            # Load dataset and iterator
            dataset = COWSL2H(DATASET_ARGS.data_dir, data_split, tokenizer, DATASET_ARGS.max_seq_length)
            loader = DataLoader(dataset, batch_size=DATASET_ARGS.train_batch_size, num_workers=4)
            it = iter(loader)

            count = 0
            for batch in it:
                preds, texts, targets = predict_batch(model, batch, num_beams=args.num_beams)

                # Output one file with original, target, and predicted combined
                # Output one file each for original, tartget, and predicted sentences
                for i in range(len(texts)):
                    f_comb.write("Original Sentence: {}\nTarget Sentence: {}\nPredicted Sentence: {}\n\n\n".format(
                        texts[i], targets[i], texts[i]
                    ))
                    f_orig.write(texts[i] + '\n')
                    f_tar.write(targets[i] + '\n')
                    f_pred.write(texts[i] + '\n')

                count += 1
                print('Processed {} batches'.format(count), end='\r')
