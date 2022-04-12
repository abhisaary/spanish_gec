import argparse

import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration

DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(text, tokenizer, model):
    tokenized_text = tokenizer.encode(text, return_tensors="pt").to(DEVICE)

    summary_ids = model.generate(
        tokenized_text,
        max_length=150,
        num_beams=5,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('text', type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')
    model = MT5ForConditionalGeneration.from_pretrained(args.model_dir)
    model.to(DEVICE)

    print(predict(args.text, tokenizer, model))
