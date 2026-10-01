"""
Step 5 - Fine-tune a small language model (DistilBERT) with a PyTorch training loop.

DistilBERT is a 66M-parameter "distilled" version of BERT: about 40% smaller and
60% faster, keeping ~97% of BERT's language understanding. It was pre-trained on
Wikipedia + books, so it already understands English. We only teach it our task
by adding a small classification head and training for 2 epochs ("fine-tuning").

Why a hand-written loop instead of Hugging Face's Trainer?
So every step is visible: batches -> forward pass -> loss -> backward pass ->
optimizer step -> learning-rate schedule. That is what PyTorch interviews ask about.

Run this on a GPU (free on Kaggle / Colab - see notebooks/). On a CPU it is very slow.
"""
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src import config
from src.data.prepare import load_split
from src.models.evaluate import compute_metrics, save_results


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class ReviewDataset(Dataset):
    """Holds raw texts + labels. Tokenisation happens per batch in the collate fn."""

    def __init__(self, texts: list[str], labels: list[int]):
        self.texts, self.labels = texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def make_collate_fn(tokenizer, max_length: int):
    """
    Tokenise a whole batch at once. padding=True pads only to the longest review
    IN THIS BATCH (not to max_length), which saves a lot of compute.
    """
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate


def make_loader(df, tokenizer, batch_size, shuffle, max_length):
    ds = ReviewDataset(df["review"].tolist(), df["label"].tolist())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=make_collate_fn(tokenizer, max_length))


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Square-root inverse frequency: rare classes (neutral) count more in the loss,
    but less aggressively than full inverse weighting (which over-predicts neutral).
    """
    counts = np.bincount(labels, minlength=len(config.LABELS)).astype(float)
    w = 1.0 / np.sqrt(counts)
    return torch.tensor(w / w.mean(), dtype=torch.float)


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model, loader, device) -> np.ndarray:
    model.eval()                        # turns off dropout
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def train_transformer(model_name: str = config.TRANSFORMER_MODEL_NAME,
                      output_dir=config.TRANSFORMER_DIR,
                      epochs: int = config.EPOCHS,
                      batch_size: int = config.BATCH_SIZE,
                      max_length: int = config.MAX_LENGTH,
                      train_subset: int | None = config.TRAIN_SUBSET,
                      save_outputs: bool = True,
                      data: dict | None = None) -> dict:
    torch.manual_seed(config.RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"     # mixed precision: ~2x faster on GPU
    print(f"  Device: {device}" + (" (mixed precision)" if use_amp else
                                  "  - WARNING: no GPU, this will be very slow"))

    data = data or {s: load_split(s) for s in ("train", "val", "test")}
    train_df, val_df, test_df = data["train"], data["val"], data["test"]
    if train_subset and train_subset < len(train_df):
        train_df = train_df.sample(train_subset, random_state=config.RANDOM_STATE)
    print(f"  Train {len(train_df):,} | Val {len(val_df):,} | Test {len(test_df):,}")

    # 1. Pre-trained tokenizer + model with a fresh 3-class head
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(config.LABELS),
        id2label=dict(enumerate(config.LABELS)),
        label2id={l: i for i, l in enumerate(config.LABELS)},
    ).to(device)

    train_loader = make_loader(train_df, tokenizer, batch_size, True, max_length)
    val_loader = make_loader(val_df, tokenizer, batch_size * 2, False, max_length)
    test_loader = make_loader(test_df, tokenizer, batch_size * 2, False, max_length)

    # 2. Loss, optimizer, learning-rate schedule
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_df["label"].to_numpy()).to(device))
    no_decay = ["bias", "LayerNorm.weight"]             # standard: don't decay these
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": config.WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=config.LEARNING_RATE)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=math.ceil(config.WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 3. Training loop
    best_f1, best_epoch, history = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()                   # turns dropout on
        start, running = time.time(), 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(**batch).logits          # forward pass
                loss = loss_fn(logits.float(), labels)  # how wrong were we?

            optimizer.zero_grad()
            scaler.scale(loss).backward()               # backward pass: gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoid exploding grads
            scaler.step(optimizer)                      # update weights
            scaler.update()
            scheduler.step()                            # adjust learning rate

            running += loss.item()
            if step % 200 == 0 or step == len(train_loader):
                print(f"    epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / step:.4f} ({time.time() - start:.0f}s)")

        # 4. Validate after every epoch, keep the best checkpoint
        val_pred = predict_proba(model, val_loader, device).argmax(1)
        val_metrics = compute_metrics(val_df["label"], val_pred)
        history.append({"epoch": epoch, "train_loss": round(running / len(train_loader), 4),
                        **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"  Epoch {epoch}: val macro-F1 {val_metrics['macro_f1']:.4f} | "
              f"val accuracy {val_metrics['accuracy']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    # 5. Final test evaluation with the BEST checkpoint
    print(f"  Best epoch: {best_epoch} (val macro-F1 {best_f1:.4f}) - evaluating on test")
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    proba = predict_proba(model, test_loader, device)
    extra = {"val_macro_f1": round(best_f1, 4), "best_epoch": best_epoch,
             "train_size": len(train_df), "epochs": epochs, "max_length": max_length,
             "base_model": model_name, "history": history}
    if save_outputs:
        metrics = save_results("transformer", "DistilBERT (fine-tuned)", test_df,
                               proba.argmax(1), proba, extra=extra)
        (output_dir / "training_summary.json").write_text(json.dumps(metrics, indent=2))
    else:
        metrics = {**compute_metrics(test_df["label"], proba.argmax(1)), **extra}
    print(f"  Test: accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f}")
    return metrics


if __name__ == "__main__":
    train_transformer()
