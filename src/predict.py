import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import PreferenceDataset, get_tokenizer
from model import PreferenceModel


def predict():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv(cfg.test_path)
    tokenizer = get_tokenizer(cfg.model_name)
    test_dataset = PreferenceDataset(test_df, tokenizer, cfg.max_length, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=2)

    model = PreferenceModel(cfg.model_name, cfg.num_labels)
    model.load_state_dict(torch.load(f"{cfg.output_dir}/best_model.pt", map_location=device))
    model.to(device).eval()

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_preds.append(torch.softmax(outputs["logits"], dim=-1).cpu().numpy())

    all_preds = np.concatenate(all_preds)
    submission = pd.DataFrame({
        "id": test_df["id"],
        "winner_model_a": all_preds[:, 0],
        "winner_model_b": all_preds[:, 1],
        "winner_tie": all_preds[:, 2],
    })
    submission.to_csv("submission.csv", index=False)
    print(submission.head())


if __name__ == "__main__":
    predict()
