import argparse

DATASET_ARGS = dict(
        data_dir="../cowsl2h/data/",
        default_root_dir="../train_outputs/",
        model_name_or_path='google/mt5-small',
        tokenizer_name_or_path='google/mt5-small',
        max_seq_length=40,
        train_batch_size=4,
        eval_batch_size=4,
        seed=42,
)
DATASET_ARGS = argparse.Namespace(**DATASET_ARGS)
