import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from config import Config
from dataset import PreferenceDataset, get_tokenizer
from model import PreferenceModel


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(batch["labels"].numpy())

    all_preds = np.concatenate(all_preds)
    true_classes = np.concatenate(all_labels).argmax(axis=1)
    return log_loss(true_classes, all_preds, labels=[0, 1, 2])


def train():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(cfg.train_path)
    train_df, val_df = train_test_split(
        df, test_size=cfg.val_split, random_state=cfg.seed, shuffle=True
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    tokenizer = get_tokenizer(cfg.model_name)
    train_dataset = PreferenceDataset(train_df, tokenizer, cfg.max_length)
    val_dataset = PreferenceDataset(val_df, tokenizer, cfg.max_length)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=2)

    model = PreferenceModel(cfg.model_name, cfg.num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    total_steps = len(train_loader) * cfg.epochs // cfg.gradient_accumulation_steps
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda") if cfg.fp16 else None

    best_val_loss = float("inf")
    os.makedirs(cfg.output_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if cfg.fp16:
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs["loss"] / cfg.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"] / cfg.gradient_accumulation_steps
                loss.backward()

            running_loss += loss.item()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if (step + 1) % (cfg.gradient_accumulation_steps * 50) == 0:
                    print(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | "
                          f"Loss: {running_loss/50:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
                    running_loss = 0.0

        val_loss = validate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Val Log Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{cfg.output_dir}/best_model.pt")
            print(f"Saved best model: {val_loss:.4f}")

    print(f"Done. Best Val Log Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
