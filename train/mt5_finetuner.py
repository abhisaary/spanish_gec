import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

from train.cowsl2h import COWSL2H


class MT5Finetuner(pl.LightningModule):
    """
    Create basic finetuner class using pytorch lightning. All that is required is defining a few training methods,
    as well as a dataset loader. This class uses the mT5 model and tokenizer from HuggingFace, and finetunes it
    via a pytorch lightning model.train() call.
    """
    def __init__(self, params):
        super(MT5Finetuner, self).__init__()
        self.save_hyperparameters()

        self.params = params
        self.model = MT5ForConditionalGeneration.from_pretrained(params.model_name_or_path)
        self.tokenizer = MT5Tokenizer.from_pretrained(params.tokenizer_name_or_path)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_train_loss)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_val_loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.params.learning_rate)

    def train_dataloader(self):
        train_dataset = COWSL2H(self.params.data_dir, "train", self.tokenizer, self.params.max_seq_length)
        return DataLoader(train_dataset, batch_size=self.params.train_batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        val_dataset = COWSL2H(self.params.data_dir, "val", self.tokenizer, self.params.max_seq_length)
        return DataLoader(val_dataset, batch_size=self.params.eval_batch_size, num_workers=4)
