import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, time, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(time, time, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, time, dim)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class FoundationLM(nn.Module):
    def __init__(self, vocab_size=4096, context_length=256, dim=320, heads=8, layers=6, hidden_dim=1280, dropout=0.1):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(context_length, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, hidden_dim, dropout)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens, targets=None):
        _, time = tokens.shape
        if time > self.context_length:
            raise ValueError("Sequence exceeds configured context length")

        positions = torch.arange(time, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.norm(x))
        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=0.9, top_k=40):
        for _ in range(max_new_tokens):
            context = tokens[:, -self.context_length:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat((tokens, next_token), dim=1)

        return tokens
