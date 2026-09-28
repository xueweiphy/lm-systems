# lm-systems

Systems for training a language model from scratch: ring all-reduce, data-parallel
training, communication benchmarks, attention memory/time benchmarks, and FlashAttention-2. Companion to [lm-from-scratch](https://github.com/xueweiphy/lm-from-scratch),
which holds the model; this repo holds the systems needed to train it at scale.
Follows CS336 Assignment 2 (sections 4 and 5).

## What's here

| folder | contents |
|---|---|
| `allreduce/` | ring all-reduce in numpy — scatter-reduce then all-gather, 2(N-1) steps — with tests against a naive sum |
| `ddp/` | `comm_benchmark.py`: times `dist.all_reduce` over message size and world size. `naive_ddp.py`: a minimal `DDP` wrapper — broadcast parameters from rank 0, all-reduce gradients after `backward()` — verified against single-process training |
| `flash/` | FlashAttention-2: `flash_attention.py` — tiled forward in PyTorch (online softmax, saves only the log-sum-exp L) and the recomputation backward (`torch.compile`); `flash_attention_triton.py` — the forward as a Triton kernel with causal masking; `test_flash.py` — forward and gradients against PyTorch's SDPA |
| `attention/` | `bench_attention.py`: forward/backward time and memory of naive causal attention (own softmax, single head, batch 8) over sequence length and head dimension |
| `results/` | benchmark data and plots |

## Result so far

gloo all-reduce on one Mac mini (10 cores, 32 GB), float32, 1 MB to 1 GB, 2/4/6 processes:

![scaling](results/allreduce_scaling.png)

Above ~10 MB every world size converges on the same **~14.5 GB/s** of aggregate
bandwidth, 2(N-1)·D / t. That is the machine's memory bandwidth, shared by all
processes — so time grows as (N-1), not as the near-flat 2(N-1)/N a real network
would give a ring. One machine is not a network; the multi-node numbers are still to come.

Below ~10 MB the cost is per-step latency, ~150-240 µs per collective. That gap is
why flat (single-buffer) DDP exists.

## Naive attention: time and memory

A100-40GB, MIG slice 3g.20gb (19.8 GiB free), fp32, batch 8, single head, 100 runs each.
Ranges are over head dimension D = 16 ... 128 (`results/attention_a100mig.csv`).

| T | forward ms | backward ms | MiB in use before backward | peak MiB |
|---|---|---|---|---|
| 256 | 0.40 | 1.0 | 13 - 24 | 29 - 35 |
| 1024 | 1.40 - 1.65 | 3.2 - 3.5 | 83 - 97 | 213 - 237 |
| 4096 | 18.1 - 22.1 | 41.8 - 46.1 | 1065 - 1121 | 3119 - 3217 |
| 8192 | 71.5 - 87.5 | 161 - 178 | 4193 - 4305 | 12397 - 12593 |
| 16384 | OOM | | | |

With S = B·T²·4 bytes (one float T×T tensor), the forward keeps **2 S** alive for backward
(the `exp` output and the attention probabilities) plus a T² byte mask; forward + backward
peaks at about **6 S**. At T = 8192 that predicts 4096 + 64 + 32 = 4192 MiB against 4193 measured.
At T = 16384, S = 8 GiB and the forward needs a third T×T tensor before it finishes, so it
runs out of memory for every D.

Time grows as T² (3.94x from 4096 to 8192) while 8x more head dimension costs only 22 %:
the elementwise T² operations, limited by memory bandwidth, dominate the matmuls. Both
results are the case for FlashAttention: save only Q, K, V and the per-row log-sum-exp,
recompute by tiles in backward, never materialise the T×T matrix.

## FlashAttention-2

Forward: one Triton program per (query tile, batch element) loops over key tiles,
keeping a running max m, sum l and unnormalised output O on chip in fp32, rescaling both
by exp(m_old − m_new) when the max moves, and dividing by l once at the end. Nothing of
size T × T is written to memory; only O and L = m + log l are saved. Causal masking uses
absolute positions and stops the key loop at the diagonal.

Backward: recompute P = exp(S − L) from the saved L and use D = rowsum(O ∘ dO), which
equals rowsum(P ∘ dP) — so no softmax Jacobian and no stored T × T activations.
Written in PyTorch and compiled with `torch.compile`, shared by both versions.

Correctness on an A100 MIG slice: forward within 2e-3 of SDPA in fp32 (the gap is
Triton's TF32 matmul; 5e-7 with IEEE fp32), one bf16 ulp (7.8e-3) in bf16; gradients
within 1–3e-3, causal or not. Passes the CS336 A2 FlashAttention tests (forward and
backward, PyTorch and Triton, causal and not). Benchmarks against naive, compiled and
SDPA attention to follow.

## Run

```
pytest allreduce/                    # ring all-reduce correctness
python ddp/comm_benchmark.py         # writes allreduce.csv to the current directory
python ddp/naive_ddp.py              # 4-process DDP; all ranks print identical weights
python attention/bench_attention.py  # needs a CUDA GPU; writes pytorch_attention.csv
pytest flash/                        # FlashAttention-2 vs SDPA; Triton tests need a CUDA GPU
```

The all-reduce and DDP parts are CPU only (gloo backend); the attention benchmark needs a GPU.
