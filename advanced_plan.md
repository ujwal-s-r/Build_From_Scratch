# Advanced LLM Track (Post-Attention)

## Table of Contents

### Module 7: Reasoning & Test-Time Compute

- 7.1 Chain-of-thought scaling laws and the third scaling axis
- 7.2 Outcome vs Process Reward Models (ORM vs PRM)
- 7.3 GRPO — Group Relative Policy Optimization (critic-free RL)
- 7.4 The R1-Zero recipe: pure RL, emergent reasoning, "aha moments"
- 7.5 Cold-start SFT + multi-stage RL (full R1 pipeline)
- 7.6 Self-consistency, beam search, and MCTS over reasoning chains
- 7.7 Adaptive test-time compute allocation (spend more only on hard problems)

### Module 8: Post-Transformer & Hybrid Architectures

- 8.1 Recap: where Mamba-2 hits its ceiling (state-tracking, hardware utilization)
- 8.2 Mamba-3: exponential-trapezoidal discretization, complex state updates
- 8.3 MIMO (multi-input multi-output) formulation for decode throughput
- 8.4 Data-dependent RoPE trick for state tracking
- 8.5 Linear-RNN family survey: RWKV-7, RetNet, Griffin, xLSTM
- 8.6 Hybrid architectures: Jamba, Nemotron-3-Super (SSM + attention blocks)
- 8.7 Mixture-of-Experts internals: routing, load balancing, DeepSeek-V3

### Module 9: Model Compression — Quantization

- 9.1 Precision ladder: FP32 → BF16 → INT8 → INT4 → INT2
- 9.2 Post-training quantization (PTQ) vs quantization-aware training (QAT)
- 9.3 The 5% accuracy-drop rule of thumb for choosing compression ratio
- 9.4 KV-cache quantization (building on your existing KV-cache module)

### Module 10: Model Compression — Distillation & Pruning

- 10.1 Classic KD: soft-label / distribution matching (teacher logits vs hard labels)
- 10.2 On-policy distillation (student generates, teacher scores)
- 10.3 RL-based distillation with LLM-as-judge (label-free)
- 10.4 Structured vs unstructured pruning; why structured wins on standard hardware
- 10.5 Combined recipe: prune 30% → distill to recover accuracy
- 10.6 Subspace-targeted distillation for domain-specific students

### Module 11: Inference-Time Efficiency

- 11.1 Speculative decoding (draft model + verifier)
- 11.2 Model cascades and routing across model sizes
- 11.3 Batching strategies and continuous batching for throughput

## Papers to Actually Implement

These are chosen because each has a compact, reproducible core algorithm you can code from scratch on a small model or toy task, matching your existing style (TinyShakespeare-scale experiments).

| Paper | What to implement | Why it's tractable |
| --- | --- | --- |
| DeepSeek-R1 (arXiv:2501.12948) — pure RL reasoning via GRPO | GRPO loop on a tiny model + verifiable reward (e.g., arithmetic/simple code correctness) | The recipe is described as "insensitive to implementation details" — works with small base models, group size 16-128, no critic model needed |
| Mamba-3 (arXiv:2603.15569) — trapezoidal discretization + MIMO | Modify your existing Mamba module: swap Euler discretization for trapezoidal, add complex state updates | Kernels are open-sourced; builds directly on your 4_attention_killer_SSM_mamba code |
| RL-based Knowledge Distillation with LLM-as-Judge (arXiv:2604.02621) | Distill a small model using single-token LLM-judge rewards instead of labeled data | Judge reward is single-token, so it's cheap to compute — good fit for your token-efficiency focus |
| MiniLLM: On-Policy Distillation (arXiv:2306.08543) | Replace forward-KL with reverse-KL distillation objective, train student on its own generations | Well-established baseline, lots of reference implementations exist to check your work |
| SubDistill (arXiv:2601.05913) | Layer-wise distillation targeting only relevant subspaces for a narrow task | Good match for your entity-resolution/domain-specific use cases — smaller scope than full-model KD |
| Knowledge Distillation for Time Series Classification (arXiv:2607.06796) | Teacher-student on a non-LLM task (FCN/Inception/ConvTran) to isolate the distillation mechanics from LLM complexity | Comes with public code (GitHub linked in paper) — good first KD implementation before tackling LLMs |

## Suggested Build Sequence