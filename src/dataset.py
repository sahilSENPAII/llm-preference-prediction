import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def smart_truncate(prompt, response_a, response_b, tokenizer, max_length):
    special_tokens_count = 4
    available = max_length - special_tokens_count
    prompt_budget = available // 4
    response_budget = (available - prompt_budget) // 2

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)[:prompt_budget]
    resp_a_tokens = tokenizer.encode(response_a, add_special_tokens=False)[:response_budget]
    resp_b_tokens = tokenizer.encode(response_b, add_special_tokens=False)[:response_budget]

    prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    resp_a_text = tokenizer.decode(resp_a_tokens, skip_special_tokens=True)
    resp_b_text = tokenizer.decode(resp_b_tokens, skip_special_tokens=True)

    return f"[Prompt]: {prompt_text} [Response A]: {resp_a_text} [Response B]: {resp_b_text}"


class PreferenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = smart_truncate(
            str(row["prompt"]),
            str(row["response_a"]),
            str(row["response_b"]),
            self.tokenizer,
            self.max_length,
        )

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if not self.is_test:
            item["labels"] = torch.tensor(
                [row["winner_model_a"], row["winner_model_b"], row["winner_tie"]],
                dtype=torch.float,
            )

        return item
