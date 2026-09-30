import sys
import json
from pathlib import Path
import torch

from foundationlm.model import FoundationLM


def main():
    if len(sys.argv) != 2:
        print("Usage: python training/generate.py model.pt")
        return

    config = json.loads(Path("configs/foundationlm_v03.json").read_text(encoding="utf-8"))
    model = FoundationLM(**config["model"])
    model.load_state_dict(torch.load(sys.argv[1], map_location="cpu"))
    model.eval()

    seed = torch.tensor([[1]], dtype=torch.long)
    output = model.generate(seed, max_new_tokens=80)
    print(output[0].tolist())


if __name__ == "__main__":
    main()
