"""Reward model for RLHF: a small model with a 1-number "score" head, trained on human preferences
(question, chosen, rejected) so that score(chosen) > score(rejected). Then RL uses it as the reward.

Stored as a LoRA adapter (+ score head) on the base model, kind "rm" in the registry.
"""
import json
import random
from pathlib import Path

from . import model_loader, registry


def split(rows, val_frac=0.2, seed=42):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(round(len(rows) * val_frac))) if len(rows) >= 5 else 0
    return rows[n_val:], rows[:n_val]


def text(tok, prompt, answer):
    """Exactly the text the reward model scores: chat-formatted question + answer + end token."""
    return model_loader.format_prompt(tok, prompt) + str(answer).strip() + tok.eos_token


def to_dataset(tok, rows):
    from datasets import Dataset
    return Dataset.from_list([{"prompt": model_loader.format_prompt(tok, r["prompt"]),
                               "chosen": str(r["chosen"]).strip() + tok.eos_token,
                               "rejected": str(r["rejected"]).strip() + tok.eos_token} for r in rows])


def accuracy(scores_chosen, scores_rejected):
    pairs = list(zip(scores_chosen, scores_rejected))
    if not pairs:
        return None, None
    acc = sum(c > r for c, r in pairs) / len(pairs)
    margin = sum(c - r for c, r in pairs) / len(pairs)
    return acc, margin


class Scorer:
    """Loads a trained reward model and scores texts. Used by RLHF training and the dashboard."""

    def __init__(self, rm_id, device=None):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification
        m = registry.model(rm_id)
        if not m or m["kind"] != "rm":
            raise model_loader.LineageError(f"{rm_id} is not a reward model.")
        path = Path(m["adapter_path"])
        if not (path / "adapter_config.json").exists():
            raise model_loader.LineageError(f"Reward model files missing at {path}.")
        self.rm_id, self.device = rm_id, device or model_loader.pick_device()
        self.tok = model_loader.load_tokenizer(m["base_model"])
        self.tok.padding_side = "right"
        base = AutoModelForSequenceClassification.from_pretrained(
            m["base_model"], num_labels=1, dtype=model_loader.pick_dtype(self.device))
        base.config.pad_token_id = self.tok.pad_token_id
        self.model = PeftModel.from_pretrained(base, str(path)).to(self.device).eval()
        self.torch = torch

    def score_texts(self, texts, batch=8, max_length=384):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True,
                               max_length=max_length).to(self.device)
                out += self.model(**enc).logits[:, 0].float().tolist()
        return out

    def score(self, prompts, answers):
        return self.score_texts([text(self.tok, p, a) for p, a in zip(prompts, answers)])

    def pair_accuracy(self, rows):
        c = self.score([r["prompt"] for r in rows], [r["chosen"] for r in rows])
        r_ = self.score([r["prompt"] for r in rows], [r["rejected"] for r in rows])
        acc, margin = accuracy(c, r_)
        return acc, margin, [{"id": x["id"], "prompt": x["prompt"], "chosen_score": round(a, 3),
                              "rejected_score": round(b, 3), "correct": a > b} for x, a, b in zip(rows, c, r_)]


def info(rm_id):
    m = registry.model(rm_id)
    return json.loads(m["reference_json"] or "{}") if m else {}
