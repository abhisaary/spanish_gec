import os

import pandas as pd
from torch.utils.data import Dataset


class COWSL2H(Dataset):
    def __init__(self, data_dir, data_split, tokenizer, max_seq_length, prefix=None, max_perc_not_corrected=None):

        self.data = pd.read_csv(os.path.join(data_dir, data_split + '.tsv'), sep='\t')
        self.prefix = prefix
        self.max_seq_length = max_seq_length
        self.max_perc_not_corrected = max_perc_not_corrected
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):

        self.data['corrected'] = self.data['input_text'] != self.data['target_text']
        max_not_cor = len(self.data)
        if self.max_perc_not_corrected is not None:
            max_not_cor *= self.max_perc_not_corrected

        not_cor = 0
        for idx, row in self.data.iterrows():
            if not row['corrected']: not_cor += 1
            if not_cor >= max_not_cor: break

            input_text, output_text = row['input_text'], row['target_text']
            if self.prefix: input_text = '{}: {}'.format(self.prefix, input_text)

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_text], max_length=self.max_seq_length, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [output_text], max_length=self.max_seq_length, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
