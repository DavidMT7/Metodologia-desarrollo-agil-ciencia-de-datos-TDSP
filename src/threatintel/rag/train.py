# -*- coding: utf-8 -*-
"""Afinado del bi-encoder con MultipleNegativesRankingLoss (InfoNCE)."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_architecture(model) -> str:
    lines = [
        "Bi-encoder siamés: la misma torre codifica la consulta y el documento.",
        "Similitud: coseno de embeddings L2-normalizados.",
        "Pérdida: MultipleNegativesRankingLoss (InfoNCE) con scale=20 (temperatura 0.05).",
        "Negativos: el resto del lote; no hay negativos etiquetados a mano.",
        "El texto indexado sale de los parquets hardened: táctica enriquecida y detection STIX.",
        "Módulos:",
    ]
    for name, module in model.named_children():
        lines.append(f"- {name}: {module.__class__.__name__}")
    config = None
    for module in model.modules():
        auto = getattr(module, "auto_model", None)
        if auto is not None:
            config = getattr(auto, "config", None)
            break
    if config is not None:
        lines.append(
            f"Transformer: {config.__class__.__name__}, capas={getattr(config, 'num_hidden_layers', '?')}, "
            f"hidden={getattr(config, 'hidden_size', '?')}, atención={getattr(config, 'num_attention_heads', '?')}."
        )
    for module in model:
        if module.__class__.__name__ == "Pooling":
            lines.append(f"Pooling: {getattr(module, 'pooling_mode', None)}.")
    lines.append(f"max_seq_length={getattr(model, 'max_seq_length', 'na')}.")
    return "\n".join(lines)


def warmup_step_count(n_pairs: int, batch_size: int, epochs: int, warmup_ratio: float) -> int:
    steps_per_epoch = max(1, int(np.ceil(n_pairs / batch_size)))
    if warmup_ratio <= 0:
        return 0
    return max(1, int(round(steps_per_epoch * epochs * warmup_ratio)))


def train_biencoder(
    model_name: str,
    train_pairs: pd.DataFrame,
    *,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    output_dir: Path,
    max_seq_length: int = 256,
    warmup_ratio: float = 0.1,
    loss_scale: float = 20.0,
    weight_decay: float = 0.0,
    lr_scheduler_type: str = "linear",
):
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.sentence_transformer.model_card import SentenceTransformerModelCardCallback

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    columns = {
        "anchor": train_pairs["query"].astype(str).tolist(),
        "positive": train_pairs["positive_text"].astype(str).tolist(),
    }
    if "negative" in train_pairs.columns:
        columns["negative"] = train_pairs["negative"].astype(str).tolist()
    dataset = Dataset.from_dict(columns)
    expected = ["anchor", "positive"] + (["negative"] if "negative" in columns else [])
    if list(dataset.column_names)[: len(expected)] != expected:
        raise RuntimeError(f"Orden de columnas inesperado: {dataset.column_names}")
    loss = losses.MultipleNegativesRankingLoss(model, scale=float(loss_scale))
    use_cuda = torch.cuda.is_available()
    steps_per_epoch = max(1, int(np.ceil(len(train_pairs) / batch_size)))
    warmup_steps = warmup_step_count(len(train_pairs), batch_size, epochs, warmup_ratio)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=float(weight_decay),
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=max(1, min(10, steps_per_epoch)),
        save_strategy="no",
        report_to=[],
        seed=seed,
        fp16=use_cuda,
        dataloader_pin_memory=use_cuda,
        dataloader_num_workers=0,
        disable_tqdm=True,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)
    trainer.remove_callback(SentenceTransformerModelCardCallback)
    trainer.train()
    history = [dict(row) for row in trainer.state.log_history if "loss" in row]
    return model, history
