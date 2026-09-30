from pathlib import Path
import json
import torch

from foundationlm.model import FoundationLM


def get_batch(tokens, batch_size, context_length):
    starts = torch.randint(0, len(tokens) - context_length - 1, (batch_size,))
    x = torch.stack([tokens[i:i + context_length] for i in starts])
    y = torch.stack([tokens[i + 1:i + context_length + 1] for i in starts])
    return x, y


def main():
    config = json.loads(Path("configs/foundationlm_v03.json").read_text(encoding="utf-8"))
    data_path = Path("train_tokens.pt")

    if not data_path.exists():
        print("train_tokens.pt was not found")
        return

    tokens = torch.load(data_path, map_location="cpu").long()
    model = FoundationLM(**config["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    for step in range(1, config["training"]["steps"] + 1):
        x, y = get_batch(tokens, config["training"]["batch_size"], config["model"]["context_length"])
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % config["training"]["log_every"] == 0:
            print(f"{step} | loss {loss.item():.4f}")

        if step % config["training"]["checkpoint_every"] == 0:
            torch.save(model.state_dict(), f"checkpoint_{step}.pt")


if __name__ == "__main__":
    main()
