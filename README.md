# FoundationLM

FoundationLM is a small transformer language model project I built while learning how modern language models work beneath the surface.

Instead of using a pretrained model, I worked through the full pipeline: tokenisation, model architecture, training, evaluation, inference and CPU benchmarking.

## Final Model

| Metric | Result |
|---|---:|
| Parameters | 19,275,648 |
| Context length | 256 |
| Vocabulary size | 4,096 |
| Training tokens | 153,608,448 |
| Best checkpoint | 150,000 steps |
| Validation loss | 3.0438 |
| Test loss | 3.0382 |
| Test perplexity | 20.86 |
| Training hardware | CPU |

## What I Worked On

- transformer architecture in PyTorch
- custom BPE tokenisation
- tokenised training data
- CPU training
- validation and test evaluation
- throughput benchmarking
- checkpoint-based text generation

## Layout

```text
foundation-lm/
├── foundationlm/
│   ├── model.py
│   └── tokenizer.py
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
├── configs/
│   └── foundationlm_v03.json
├── benchmarks/
│   └── cpu_scaling.csv
├── results/
│   ├── final_metrics.json
│   └── training_summary.txt
├── pyproject.toml
└── README.md
```

## CPU Scaling

| Candidate | Parameters | Tokens/sec |
|---|---:|---:|
| A | 2,557,632 | 6,501 |
| B | 5,868,800 | 3,150 |
| C | 11,228,480 | 1,956 |
| D | 19,275,648 | 1,174 |

The final training run used Candidate D.

## Install

```bash
pip install -e .
```

## Train

```bash
python training/train.py
```

## Evaluate

```bash
python training/evaluate.py model.pt
```

## Generate

```bash
python training/generate.py model.pt
```

## What I Learned

This project taught me about model size, memory, throughput, training time, tokenisation, validation, checkpointing and the trade-offs involved in training on limited compute.

## Author

Mike Junior Maringa  
BSc Computer Science for Cyber Security  
Oxford Brookes University
