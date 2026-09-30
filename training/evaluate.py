import sys
import json
from pathlib import Path
import torch

from foundationlm.model import FoundationLM


def main():
    if len(sys.argv) != 2:
        print("Usage: python training/evaluate.py model.pt")
        return

    config = json.loads(Path("configs/foundationlm_v03.json").read_text(encoding="utf-8"))
    data_path = Path("test_tokens.pt")

    if not data_path.exists():
        print("test_tokens.pt was not found")
        return

    model = FoundationLM(**config["model"])
    model.load_state_dict(torch.load(sys.argv[1], map_location="cpu"))
    model.eval()

    tokens = torch.load(data_path, map_location="cpu").long()
    context = config["model"]["context_length"]
    losses = []

    with torch.no_grad():
        for start in range(0, len(tokens) - context - 1, context):
            x = tokens[start:start + context].unsqueeze(0)
            y = tokens[start + 1:start + context + 1].unsqueeze(0)
            _, loss = model(x, y)
            losses.append(loss.item())

    average = sum(losses) / len(losses)
    perplexity = torch.exp(torch.tensor(average)).item()
    print(f"Loss: {average:.4f}")
    print(f"Perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    main()
