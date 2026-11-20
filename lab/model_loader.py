"""Load the base model (+ optional LoRA adapter) and generate answers.

Lineage rule: every adapter in this lab is a LoRA adapter on the SAME base model. A child stage loads its
parent's adapter weights and keeps training them, then saves a new adapter. So loading
base + one adapter reproduces any version exactly; adapters are never treated as standalone models.
"""
import json
from pathlib import Path

from . import registry

SYSTEM_PROMPT = ("You are a tool store assistant. Answer store staff in one or two short sentences. "
                 "Never promise stock, dates or discounts without checking.")


class LineageError(RuntimeError):
    pass


def pick_device():
    import torch
    if torch.cuda.is_available():
        try:
            (torch.ones(1, device="cuda") + 1).item()
            return "cuda"
        except Exception:  # noqa: BLE001
            pass
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device):
    import torch
    # fp32 everywhere except GPUs with fast bf16: Pascal (GTX 10xx) fp16 math is very slow, CPU fp16 unsupported.
    if device == "cuda" and torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    return torch.float32


def format_prompt(tokenizer, question: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\nQuestion: {question}\nAnswer:"


def load_tokenizer(base_model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(base_model, device=None):
    from transformers import AutoModelForCausalLM
    device = device or pick_device()
    m = AutoModelForCausalLM.from_pretrained(base_model, dtype=pick_dtype(device))
    return m.to(device)


def base_revision(model):
    return getattr(model.config, "_commit_hash", None)


def resolve(model_id):
    """Return (base_model_name, adapter_path or None) for a registry model id. Fails loudly, never falls back."""
    m = registry.model(model_id)
    if not m:
        raise LineageError(f"Model {model_id} is not in the registry.")
    if m["deleted"]:
        raise LineageError(f"Model {model_id} was deleted.")
    if m["kind"] == "base":
        return m["base_model"], None
    path = Path(m["adapter_path"] or "")
    if not (path / "adapter_config.json").exists():
        raise LineageError(f"Adapter files for {model_id} are missing at {path}. Restore them or pick another parent.")
    cfg = json.loads((path / "adapter_config.json").read_text())
    if cfg.get("base_model_name_or_path") not in (m["base_model"], None):
        raise LineageError(f"Adapter {model_id} was trained on {cfg.get('base_model_name_or_path')} "
                           f"but the registry says {m['base_model']}.")
    return m["base_model"], str(path)


def load_for_training(parent_id, lora_cfg):
    """Load the parent as a trainable PEFT model. Base parent -> fresh LoRA; adapter parent -> continue its LoRA."""
    from peft import LoraConfig, PeftModel, get_peft_model
    base_name, adapter = resolve(parent_id)
    base = load_base(base_name)
    rev = base_revision(base)
    if adapter is None:
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **lora_cfg))
    else:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    return model, base_name, rev


class Engine:
    """Holds one base model in memory and swaps LoRA adapters on it for inference."""

    def __init__(self, base_model):
        self.base_model = base_model
        self.device = pick_device()
        self.tok = load_tokenizer(base_model)
        self.tok.padding_side = "left"
        self.model = load_base(base_model, self.device)
        self.model.eval()
        self.loaded = set()
        self.revision = base_revision(self.model)

    def _activate(self, model_id):
        from peft import PeftModel
        base_name, adapter = resolve(model_id)
        if base_name != self.base_model:
            raise LineageError(f"{model_id} uses base {base_name}; the loaded engine uses {self.base_model}.")
        if adapter is None:
            return None
        name = model_id.replace("-", "_")
        if not isinstance(self.model, PeftModel):
            self.model = PeftModel.from_pretrained(self.model, adapter, adapter_name=name)
        elif name not in self.loaded:
            self.model.load_adapter(adapter, adapter_name=name)
        self.loaded.add(name)
        self.model.set_adapter(name)
        self.model.eval()
        return name

    def generate(self, model_id, questions, max_new_tokens=96, temperature=0.0, seed=0):
        import contextlib

        import torch
        from peft import PeftModel
        from .rewards import clean_answer
        name = self._activate(model_id)
        ctx = self.model.disable_adapter() if (name is None and isinstance(self.model, PeftModel)) else contextlib.nullcontext()
        prompts = [format_prompt(self.tok, q) for q in questions]
        out = []
        torch.manual_seed(seed)
        with ctx, torch.no_grad():
            for i in range(0, len(prompts), 8):
                batch = self.tok(prompts[i:i + 8], return_tensors="pt", padding=True).to(self.device)
                kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
                kw.update(dict(do_sample=True, temperature=temperature, top_p=0.95) if temperature > 0 else dict(do_sample=False))
                gen = self.model.generate(**batch, **kw)
                for g in gen[:, batch["input_ids"].shape[1]:]:
                    out.append(clean_answer(self.tok.decode(g, skip_special_tokens=True)))
        return out


    def chat(self, model_id, messages, tools=None, max_new_tokens=256, temperature=0.0, seed=0):
        """Full chat generation: messages = [{"role": ..., "content": ...}, ...] (OpenAI style, may include
        assistant tool_calls and role="tool" results). Returns raw text (tool-call tags kept, <think> removed)."""
        import contextlib

        import torch
        from peft import PeftModel
        from .rewards import clean_answer
        name = self._activate(model_id)
        ctx = self.model.disable_adapter() if (name is None and isinstance(self.model, PeftModel)) else contextlib.nullcontext()
        kw = dict(tokenize=False, add_generation_prompt=True)
        if tools:
            kw["tools"] = tools
        if getattr(self.tok, "chat_template", None):
            try:
                prompt = self.tok.apply_chat_template(messages, enable_thinking=False, **kw)
            except TypeError:
                prompt = self.tok.apply_chat_template(messages, **kw)
        else:
            prompt = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages) + "\nassistant:"
        torch.manual_seed(seed)
        with ctx, torch.no_grad():
            batch = self.tok([prompt], return_tensors="pt").to(self.device)
            gk = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
            gk.update(dict(do_sample=True, temperature=temperature, top_p=0.95) if temperature > 0 else dict(do_sample=False))
            g = self.model.generate(**batch, **gk)[0, batch["input_ids"].shape[1]:]
        return clean_answer(self.tok.decode(g, skip_special_tokens=True))


def adapter_size(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file() and "checkpoint" not in str(f))
