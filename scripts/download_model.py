"""Download the base model into the Hugging Face cache (one-time; needs internet)."""
import sys

from huggingface_hub import snapshot_download

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
path = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt"])
print("Downloaded", model, "to", path)
