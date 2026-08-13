# Comprehensive LLM Engineering & Deep Learning Implementation Portfolio
> **Purpose**: A comprehensive, zero-information-loss technical record of implementations, mathematical formulations, system architectures, research paper replications, and performance findings across LLM pre-training, fine-tuning, inference serving, and reinforcement learning alignment. Tailored for resume updates, technical portfolio showcases, and engineering interviews.

---

## Executive Resume Summary & Key Competencies

### Specialized Technical Skill Matrix
- **Core Frameworks & Tools**: PyTorch, CUDA, Triton, FastAPI, PyTorch Profiler, NVIDIA Nsight Systems (`nsys`), SymPy, Hugging Face Transformers/Accelerate, NumPy, SciPy, Matplotlib, Seaborn.
- **LLM Core Architecture Design**: BPE Tokenizers, Absolute & Rotary Positional Embeddings (RoPE), LayerNorm & RMSNorm, Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Head Latent Attention (DeepSeek MLA), SwiGLU FFN, Decoder-only Transformer Blocks, State Space Models (SSM/Mamba), Mixture-of-Experts (MoE) Top-K Gating & Auxiliary Loss.
- **PEFT, Quantization & Model Compression**: LoRA, QLoRA (NF4, Double Quantization, Paged Optimizers), Weight-Decomposed LoRA (DoRA), S-LoRA / Punica Multi-Adapter Concurrent Serving, Uniform Affine PTQ/QAT, Dynamic Int8 KV-Cache Compression, Modern FP8 (E4M3/E5M2) & NVFP4/MXFP4 formats, Forward vs. Reverse KL Distillation, Subspace Distillation, Layer Pruning.
- **High-Performance Inference & Serving Infrastructure**: GPU Die Architecture (SMs, Tensor Cores, HBM vs. SRAM, Roofline Model), First-Principles VRAM Allocation Budgeting, Asynchronous FastAPI Streaming Servers (SSE), `torch.compile` & CUDA Graphs with Dynamic Bucket Padding, Concurrent Request Batching, TTFT / ITL / Throughput Metric Tracking Engine.
- **Distributed Training & Multi-GPU Systems**: Distributed Data Parallelism (DDP), Tensor Parallelism (Megatron-LM Column/Row Parallel), Pipeline Parallelism (1F1B schedule), ZeRO-1/2/3 Memory Sharding Mechanics.
- **Reinforcement Learning & Reasoning Alignment**: Chain-of-Thought (CoT) Scratchpads, Entropy-Gated Dynamic Budget Allocation, Outcome Reward Models (ORM) vs. Process Reward Models (PRM), Group Relative Policy Optimization (GRPO), Symbolic Math & Format Rule Engines, Multi-Stage Rejection Sampling & SFT Curation (DeepSeek-R1-Zero & DeepSeek-R1 pipeline).

---

## 1. Resume-Ready Experience Bullet Points

### LLM Architecture & Custom Model Building (From Scratch)
* **Engineered a Modular Decoder-Only Transformer Architecture from Scratch in PyTorch**, building custom implementations of BPE tokenization, Rotary Positional Embeddings (RoPE), RMSNorm, SwiGLU activations, and Grouped-Query Attention (GQA), achieving 100% architectural parity with modern LLMs (LLaMA 2/3).
* **Implemented DeepSeek Multi-Head Latent Attention (MLA)**, utilizing low-rank latent compression of Query and Key-Value states ($c_t^{KV}, c_t^Q$) paired with decoupled RoPE keys/queries to reduce KV-cache VRAM footprint during autoregressive inference while maintaining full attention expressive capacity.
* **Designed & Evaluated Next-Generation Sequence Models (SSM / Mamba & MoE)**, discretizing continuous state-space models via Zero-Order Hold (ZOH) with data-dependent selection mechanisms and implementing a hybrid SSM-Attention architecture for linear $O(N)$ long-context sequence modeling; integrated Top-$K$ soft-gating Mixture-of-Experts (MoE) with load-balancing auxiliary loss.
* **Constructed an Advanced Muon Optimizer**, implementing matrix update orthogonalization via Newton-Schulz iterations on 2D weight matrices to constrain spectral norms, stabilizing large-scale model pretraining dynamics compared to standard AdamW.

### Parameter-Efficient Fine-Tuning (PEFT) & Quantization
* **Implemented Low-Rank Adaptation (LoRA), QLoRA, and Weight-Decomposed LoRA (DoRA)** from scratch, isolating directional ($V$) and magnitude ($m$) weight updates to achieve fine-tuning performance matching full fine-tuning with $<1\%$ trainable parameters.
* **Developed Low-Precision Quantization Engines**, implementing post-training quantization (PTQ), quantization-aware training (QAT) with Straight-Through Estimators (STE), Dynamic Int8 KV-Cache quantization, and experimental benchmarks across FP8 (E4M3/E5M2) and microscaling NVFP4/MXFP4 formats.
* **Architected Concurrent Multi-Adapter Serving Solutions**, utilizing S-LoRA / Punica multi-adapter execution concepts to serve heterogeneous fine-tuned models over a single base LLM instance without VRAM replication overhead.

### Model Serving & Systems Performance Engineering
* **Built an Enterprise-Grade Production LLM Serving API using FastAPI & Asyncio**, featuring Server-Sent Events (SSE) streaming token generation, dynamic request queue management, and server lifecycle warmup protocols.
* **Optimized Serving Execution via `torch.compile` & CUDA Graphs**, implementing a dynamic shape bucket-padding strategy (fixed sequence length buckets) that eliminated GPU kernel launch overhead and prevented recompilation bottlenecks in production workloads.
* **Constructed an Automated Load-Testing & Metrics Engine**, evaluating TTFT (Time-To-First-Token), ITL (Inter-Token Latency), output throughput, and p50/p95/p99 tail latency distributions across a 4D benchmark matrix (concurrency $\times$ prompt length $\times$ output length $\times$ batch size).
* **Profiled Multi-Layer GPU Bottlenecks using PyTorch Profiler and NVIDIA Nsight Systems (`nsys`)**, placing NVTX annotations to analyze memory allocation timelines, kernel execution bounds, and CUDA context synchronization overheads.

### Reinforcement Learning & Reasoning Alignment (Paper Implementations)
* **Replicated the DeepSeek-R1 & DeepSeek-R1-Zero Training Pipelines**, implementing Group Relative Policy Optimization (GRPO) without a critic network, utilizing group-normalized advantages and clipped surrogate loss with KL divergence penalties.
* **Constructed a Symbolically-Verified Rule-Based Reward Engine**, leveraging SymPy for exact mathematical correctness verification and regular expressions for structural format tag enforcement (`<think>...</think>`), avoiding neural reward model vulnerability to reward hacking.
* **Implemented MiniLLM On-Policy Reverse-KL Distillation**, replacing legacy forward-KL distillation with an on-policy reverse-KL objective ($\mathbb{D}_{\text{KL}}(p_{\text{student}} || p_{\text{teacher}})$) using teacher-mixed sampling to prevent student policy collapse into zero-probability void regions.

---

## 2. Comprehensive Module-by-Module Technical Inventory

```
===================================================================================
REPOSITORY DIRECTORY MAP & TECHNICAL BREAKDOWN
===================================================================================
├── Build_From_Scratch/
│   ├── 0_Byte_pair_encoder/         -> Custom Byte-Pair Encoding (BPE) Tokenizer
│   ├── 1_Embedding/                -> Absolute & Rotary Positional Embeddings (RoPE)
│   ├── 2_Normalization/            -> LayerNorm vs. RMSNorm Mechanics & Math
│   ├── 3_Attention/                -> Single-Head, MHA, GQA, KV-Cache, DeepSeek MLA
│   ├── 4_attention_killer_SSM_mamba/ -> Selective SSM & Mamba State-Space Models
│   ├── 5_FeedForwardNetwork/        -> SwiGLU / GeGLU FFN Implementations
│   ├── 6_TinyShakespear_LLM/        -> Full Decoder-Only LLM & Autoregressive Decoder
│   ├── 7_Resoning/                 -> CoT, Dynamic Entropy Allocation, GRPO Alignment
│   ├── 9_advanced_STUFF/            -> Muon Matrix-Orthogonalizing Optimizer
│   ├── 11_SSM_and_MOE/ & SSM_and_MOE/-> Hybrid SSM-Attention & MoE Gating
│   ├── 12_Quantization_idea/        -> PTQ, QAT, Int8 KV-Cache, FP8, NVFP4/MXFP4
│   ├── 13_distillation/            -> Forward vs. Reverse KL, Subspace Distillation
│   ├── 14_Fine-tuning/             -> LoRA, QLoRA, DoRA, Multi-Adapter Serving
│   ├── Attention_is_all_you_need/  -> Standard Transformer Reference Implementation
│   └── NeuralNetworks/             -> Autograd, Backprop, Neuron & MLP Foundations
├── Inference-and-model-serving/
│   ├── GPU_101.ipynb               -> GPU Microarchitecture, SMs, Memory Hierarchy
│   ├── base_inference/             -> VRAM Math, Async Server, CUDA Graphs, Padding
│   ├── torch-parellel/             -> DDP, Tensor Parallel, Pipeline Parallel, ZeRO
│   └── Metrics-tracker-engine/     -> TTFT/ITL Engine, PyTorch Profiler, Nsight Systems
└── research-paper-imlementations/
    ├── DeepSeek-R1.../             -> Full GRPO RL Pipeline, Reward Engine, Multi-Stage SFT
    └── MiniLLM.../                 -> On-Policy Reverse-KL LLM Distillation Engine
===================================================================================
```

---

### Module 1: Build From Scratch (`Build_From_Scratch`)

#### 1.1 Byte Pair Encoder (BPE Tokenizer) (`0_Byte_pair_encoder/tokenizer.ipynb`)
- **Implemented Mechanics**:
  - Raw byte string encoding using UTF-8 representation.
  - Iterative pair frequency counting across consecutive byte sequences.
  - Vocabulary expansion via merge rule dictionary generation (`(byte1, byte2) -> new_token_id`).
  - Regex-based pre-tokenization splitting (matching punctuation, whitespace, and alphanumeric clusters).
  - Special token injection (`<|endoftext|>`, `<|pad|>`) and OOV subword fallback handling.
- **Key Takeaways & Formulas**:
  - BPE balances vocabulary size $V$ against sequence length $L$. Compression Ratio: $\text{CR} = \frac{\text{Bytes in Raw Text}}{\text{Number of Tokens Generated}}$.
  - Avoids out-of-vocabulary crashes by grounding the base vocabulary in 256 byte primitives.

#### 1.2 Embedding & Positional Encodings (`1_Embedding/`)
- **Files**: `0_absolute_embedding.ipynb`, `1_RoPE.ipynb`, `rope.html`
- **Implemented Mechanics**:
  - Absolute Learnable & Sinusoidal Positional Embeddings.
  - Rotary Positional Embeddings (RoPE) operating directly on Query and Key vectors in 2D vector pairs.
  - Complex number rotation representation and 2D Givens rotation matrix application.
  - Interactive visual verification tool (`rope.html`).
- **Mathematical Formulations**:
  - Frequency computation for dimension index $i \in [1, d/2]$:
    $$\theta_i = 10000^{-\frac{2(i-1)}{d}}$$
  - Rotation matrix applied to 2D sub-vector $\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$ at position $m$:
    $$R_{\Theta, m}^{(i)} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$
  - Property of relative position inner-product preservation:
    $$\langle R_{\Theta, m} q, R_{\Theta, n} k \rangle = q^T R_{\Theta, n-m} k$$
- **Key Takeaways**:
  - RoPE naturally decays attention weights as distance $|m - n|$ increases without requiring explicit bias matrices.

#### 1.3 Normalization Mechanics (`2_Normalization/0_norm.ipynb`)
- **Implemented Mechanics**:
  - Layer Normalization (LayerNorm) vs. Root Mean Square Normalization (RMSNorm).
  - Pre-LN vs. Post-LN Transformer block placement.
- **Mathematical Formulations**:
  - LayerNorm:
    $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i, \quad \mu = \frac{1}{d}\sum_{j=1}^d x_j, \quad \sigma^2 = \frac{1}{d}\sum_{j=1}^d (x_j - \mu)^2$$
  - RMSNorm:
    $$\hat{x}_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon}$$
- **Key Takeaways**:
  - RMSNorm reduces computational overhead by $7\%\text{--}10\%$ by dropping mean computation $\mu$ without compromising training stability.

#### 1.4 Attention Mechanisms & Variants (`3_Attention/`)
- **Files**: `0_single_head_attention.ipynb`, `1_multi_head_attention.ipynb`, `2_KV_cache.ipynb`, `3_grouped_Q_A.ipynb`, `3_grouped_Q_A(LLama).ipynb`, `4_MH_latent_attention(deepseek).ipynb`, `mla_explorer.html`
- **Implemented Mechanics**:
  - **Single & Multi-Head Attention (MHA)**: Causal masking, scaled dot-product attention ($\text{Softmax}(QK^T / \sqrt{d_k})V$).
  - **Key-Value (KV) Cache**: Dynamic allocation and update of KV buffers during autoregressive generation, eliminating redundant prompt tensor re-computations.
  - **Grouped-Query Attention (GQA)**: Partitioning query heads into $G$ groups sharing a single Key/Value head pair (LLaMA 2/3 standard).
  - **Multi-Head Latent Attention (DeepSeek MLA)**: Low-rank latent compression of Query states ($c_t^Q$) and Key-Value states ($c_t^{KV}$), paired with decoupled RoPE keys/queries ($k_t^R, q_{t,i}^R$).
- **Mathematical Formulations (DeepSeek MLA)**:
  - Compressed KV Latent Vector: $c_t^{KV} = W^{DKV} h_t \quad (c_t^{KV} \in \mathbb{R}^{d_c})$
  - De-compressed Keys & Values: $k_t^C = W^{UK} c_t^{KV}, \quad v_t^C = W^{UV} c_t^{KV}$
  - Decoupled RoPE Key: $k_t^R = \text{RoPE}(W^{KR} h_t)$
  - Final Key concatenation: $k_{t,i} = \begin{bmatrix} k_t^C \\ k_t^R \end{bmatrix}$
- **Key Takeaways & Memory Impact**:
  - Standard MHA KV-Cache size per token: $2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times b_{\text{bytes}}$.
  - GQA reduces KV cache size by factor of $\frac{n_{\text{heads}}}{n_{\text{kv\_heads}}}$ (typically $4\times\text{--}8\times$).
  - MLA compresses generation VRAM requirements even further while maintaining full multi-head expressiveness.

#### 1.5 State Space Models (SSM/Mamba) & MoE (`4_attention_killer_SSM_mamba/`, `11_SSM_and_MOE/`, `SSM_and_MOE/`)
- **Files**: `0_intro.ipynb`, `1_code.ipynb`, `0_basic.ipynb`, `1_hybrid_vs_pureAttention.ipynb`, `3_data_dep_RoPE.ipynb`
- **Implemented Mechanics**:
  - **Continuous-to-Discrete State Space Models**: Continuous-time state transition discretized via Zero-Order Hold (ZOH).
  - **Selective State-Space (Mamba)**: Input-dependent state parameters $\Delta(x), B(x), C(x)$.
  - **Hybrid SSM-Attention Architecture**: Alternating SSM layers (selective memory) and Attention layers (exact long-range retrieval).
  - **Mixture-of-Experts (MoE)**: Top-$K$ Gating router, expert layer execution, and auxiliary load balancing loss.
- **Mathematical Formulations**:
  - Discretization via ZOH:
    $$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$
  - Recurrent Update: $h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t$
  - MoE Load Balancing Auxiliary Loss:
    $$\mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{i=1}^N f_i P_i$$
    where $f_i$ is the fraction of tokens routed to expert $i$, and $P_i$ is the average routing probability assigned to expert $i$.

#### 1.6 Feed-Forward Networks (FFN) (`5_FeedForwardNetwork/inti.ipynb`)
- **Implemented Mechanics**:
  - Standard MLP vs. SwiGLU / GeGLU activations.
- **Mathematical Formulations**:
  $$\text{SwiGLU}(x) = \left( \text{Swish}(x W_{\text{gate}}) \odot x W_{\text{up}} \right) W_{\text{down}}$$
  where $\text{Swish}(z) = z \cdot \sigma(z)$.
- **Key Takeaways**:
  - SwiGLU improves convergence rates and downstream task accuracy; standard dimension hidden scaling uses $\frac{8}{3} d_{\text{model}}$ to match MLP parameter counts.

#### 1.7 TinyShakespeare LLM & Decoder Architecture (`6_TinyShakespear_LLM/`)
- **Files**: `0_tiny_model.ipynb`, `1_tiny_llama.ipynb`
- **Implemented Mechanics**:
  - Full assembly of decoder-only transformer blocks.
  - Training loop on Shakespeare text corpus: Cross-entropy loss, AdamW optimizer, cosine learning rate scheduler with linear warmup.
  - Autoregressive inference engine supporting Temperature scaling, Top-$K$, and Nucleus (Top-$P$) sampling.

#### 1.8 Reasoning Engines & Alignment Mechanics (`7_Resoning/`)
- **Files**: `0_base.ipynb`, `1_entropy_regulation.ipynb`, `2_policy_reward_types.ipynb`, `3_GRPO.ipynb`, `4_RL.ipynb`, `5_training.ipynb`
- **Implemented Mechanics**:
  - **Chain-of-Thought Scratchpad**: Formatting traces inside `<think>...</think>` tags.
  - **Entropy-Gated Dynamic Budget Allocator**: Real-time evaluation of token prediction entropy $H(p_t) = -\sum p_t \log p_t$. When entropy exceeds a high threshold (uncertainty spike), allocation of additional reasoning/thinking budget is triggered dynamically.
  - **Outcome vs. Process Reward Models (ORM vs. PRM)**: Analysis of credit assignment bottlenecks.
  - **Group Relative Policy Optimization (GRPO)**: Critic-free RL algorithm normalizing rewards within sampled response groups $G$.

#### 1.9 Advanced Optimizers (`9_advanced_STUFF/0_muon_optimizer.ipynb`)
- **Implemented Mechanics**:
  - Custom **Muon Optimizer** (Matrix Update Orthogonalization via Newton-Schulz iterations).
  - Iterative update to force 2D weight update matrices toward orthogonal matrices ($O^T O = I$).
- **Mathematical Formulation (Newton-Schulz Iteration)**:
  $$X_0 = \frac{G}{\|G\|_F}, \quad X_{k+1} = \frac{1}{2} X_k \left( 3 I - X_k^T X_k \right)$$
- **Key Takeaways**:
  - Prevents feature collapse in internal 2D Linear projection layers and stabilizes high-learning-rate regimes.

#### 1.10 Quantization Foundations & Advanced Formats (`12_Quantization_idea/`)
- **Files**: `base.ipynb`, `post_train_Q.ipynb`, `QAT.ipynb`, `quantization_metrics.ipynb`, `k-v-cache-int8.ipynb`, `FP.ipynb`, `NVFP4_MXFP4.ipynb`
- **Implemented Mechanics**:
  - **Uniform Affine & Symmetric Quantization**: Floating-point $x \in [\alpha, \beta]$ mapped to integer $q \in [q_{\min}, q_{\max}]$.
  - **Quantization-Aware Training (QAT)**: Straight-Through Estimator (STE) passing gradients through rounding functions during backward passes.
  - **Dynamic Int8 KV-Cache Quantization**: Quantizing KV Cache vectors per-token dynamically, cutting memory bandwidth requirements by $50\%$.
  - **Microscaling Formats (FP8, NVFP4, MXFP4)**: Shared block scaling factors across 32-element vectors (E4M3 for weights, E5M2 for gradients/activations).
- **Mathematical Formulations**:
  - Scale & Zero-Point:
    $$S = \frac{\beta - \alpha}{q_{\max} - q_{\min}}, \quad Z = \text{round}\left(-\frac{\alpha}{S}\right) + q_{\min}$$
  - Quantize & De-quantize:
    $$q = \text{clamp}\left(\text{round}\left(\frac{x}{S}\right) + Z, q_{\min}, q_{\max}\right), \quad \hat{x} = S \cdot (q - Z)$$

#### 1.11 Knowledge Distillation & Model Pruning (`13_distillation/`)
- **Files**: `0_base.ipynb`, `forward_vs_reverse_KL.ipynb`, `subspace-distillation.ipynb`, `pruning_AND_distillation.ipynb`, `3_LLM_as_Judge.ipynb`
- **Implemented Mechanics**:
  - **Forward KL vs. Reverse KL**:
    - Forward KL ($\mathbb{D}_{\text{KL}}(P_{\text{teacher}} || Q_{\text{student}})$): Mean-seeking, spreads student probability mass over non-existent modes (causes hallucination).
    - Reverse KL ($\mathbb{D}_{\text{KL}}(Q_{\text{student}} || P_{\text{teacher}})$): Mode-seeking, forces student to focus strictly on sharp teacher modes.
  - **Subspace Distillation**: Linear adapter projections aligning disparate hidden dimension spaces ($d_{\text{teacher}} \neq d_{\text{student}}$).
  - **Structured Pruning + Fine-Tuning Distillation**: Dropping redundant transformer layers followed by student-teacher recovery distillation.
  - **LLM-as-a-Judge**: Evaluator prompting pipeline to benchmark model outputs.

#### 1.12 Parameter-Efficient Fine-Tuning (PEFT) (`14_Fine-tuning/`)
- **Files**: `0_base.ipynb`, `1_optimization_mechanics.ipynb`, `3_LoRA.ipynb`, `4_LoRA_layers.ipynb`, `5_QLoRA.ipynb`, `6_OOM.ipynb`, `7_PROD_Q_vs_LoRA.ipynb`, `8_ADV_LoRA.ipynb`, `9_SERVING.ipynb`
- **Implemented Mechanics**:
  - **LoRA (Low-Rank Adaptation)**: Factorizing weight updates $\Delta W = \frac{\alpha}{r} (A \cdot B)$ with rank $r \ll d$.
  - **QLoRA**: NormalFloat4 (NF4) data type, Double Quantization (DQ), Paged Optimizers utilizing CUDA Unified Memory to prevent OOM spikes.
  - **Weight-Decomposed LoRA (DoRA)**: Decoupling weight matrix into magnitude vector $m$ and directional matrix $V$:
    $$W = m \frac{V}{\|V\|_F}, \quad V = V_0 + \Delta V_{\text{LoRA}}$$
  - **Concurrent Multi-Adapter Serving**: S-LoRA & Punica kernel execution principles, enabling unified base model VRAM usage while routing dynamic adapter matrices per request.

---

### Module 2: Inference & Model Serving Systems (`Inference-and-model-serving`)

#### 2.1 GPU Microarchitecture & Memory Hierarchy (`GPU_101.ipynb`, `base_inference/0_understnading_inference.ipynb`)
- **Hardware Architecture Mapping**:
  - CPU vs. GPU paradigm: Low latency/heavy cache (CPU) vs. Massive parallelism/high bandwidth (GPU).
  - Streaming Multiprocessors (SMs), CUDA Cores, Tensor Cores, Registers, L1/L2 Cache, High Bandwidth Memory (HBM).
  - **Roofline Model Analysis**:
    - Operational Intensity $I = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$.
    - Prefill Stage: Compute-bound (matrix multiplications over large prompt token sequences).
    - Decoding Stage: Memory-bound (loading model weights and KV-cache vectors from HBM for every single generated token).

#### 2.2 First-Principles VRAM Budgeting (`base_inference/1_RAM_alloc.ipynb`)
- **Mathematical Model of GPU VRAM Footprint**:
  $$M_{\text{total}} = M_{\text{weights}} + M_{\text{KV}} + M_{\text{activations}} + M_{\text{overhead}}$$
  - Static Weights Footprint: $M_{\text{weights}} = \Phi \times b_{\text{bytes}}$ (e.g., 8B parameters in BF16 = 16 GB).
  - KV-Cache Allocation:
    $$M_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times s_{\text{seq}} \times b_{\text{batch}} \times b_{\text{bytes}}$$
  - CUDA Context & Buffer Overhead: ~500 MB - 1 GB.

#### 2.3 Production FastAPI Streaming Server & CUDA Graphs (`base_inference/`)
- **Files**: `3_base_server.ipynb`, `4_GPU.ipynb`, `5_good_practices.ipynb`, `api/main.py`, `router.py`, `inference.py`, `schemas.py`, `config.py`, `benchmark_harness.py`
- **Architectural Mechanics**:
  - **Asynchronous Architecture**: Non-blocking Python `asyncio` loop handling incoming HTTP requests. Server-Sent Events (SSE) streaming responses token-by-token via async generators.
  - **CUDA Graphs & `torch.compile`**: Eliminates CPU kernel launch latency by pre-recording kernel invocation sequences directly onto GPU memory.
  - **Bucket Padding Strategy**: Solves "CUDA Graph Recompilation Hell" caused by dynamic sequence lengths by creating fixed sequence length buckets (e.g., [64, 128, 256, 512, 1024]). Requests are padded to the nearest bucket, avoiding graph invalidation.
  - **Server Warmup Protocol**: Pre-executing dummy inference runs across all bucket shapes during startup (`main.py` lifespan event) to pre-compile CUDA kernels before handling client traffic.

#### 2.4 Distributed Parallelism Mechanics (`torch-parellel/0_base.ipynb`)
- **Implemented Principles & Math**:
  - **Distributed Data Parallelism (DDP)**: Replicating model weights across $N$ GPUs and synchronizing gradients via AllReduce operations.
  - **Tensor Parallelism (TP)**: Megatron-LM column and row linear layer sharding:
    - Column Parallelism (Q, K, V projections): $Y_i = X W_i \quad \rightarrow \text{Concat}(Y_1, \dots, Y_k)$.
    - Row Parallelism (Output projection): $Z_i = X_i W_i \quad \rightarrow \text{AllReduce-Sum}(Z_1 + \dots + Z_k)$.
  - **Pipeline Parallelism (PP)**: Partitioning network layers sequentially across devices with 1F1B (One Forward, One Backward) micro-batch scheduling.
  - **ZeRO Sharding (Zero Redundancy Optimizer)**:
    - ZeRO-1: Sharding Optimizer States ($4\times$ memory reduction).
    - ZeRO-2: Sharding Optimizer States + Gradients ($8\times$ memory reduction).
    - ZeRO-3: Sharding Optimizer States + Gradients + Model Parameters (Linear memory reduction with GPU count).

#### 2.5 Metrics Tracker & GPU Profiler Engine (`Metrics-tracker-engine/`)
- **Files**: `0_base.ipynb`, `1_GPU_profiler.ipynb`
- **Key Metrics Formulated**:
  - **Time-To-First-Token (TTFT)**: Latency of prefill phase.
  - **Inter-Token Latency (ITL) / TPOT**: Per-token generation latency.
  - **Output Throughput**: Total generated tokens per second across client streams.
- **Profiling Stack**:
  - `torch.profiler` integration with warmup schedules to profile PyTorch operators.
  - NVIDIA Nsight Systems (`nsys`) annotations via `torch.cuda.nvtx.range_push()` and `range_pop()`, mapping CUDA kernel launch bottlenecks to Python stack frames.

---

### Module 3: Research Paper Implementations (`research-paper-imlementations`)

#### 3.1 DeepSeek-R1: Reinforcement Learning for LLM Reasoning Capability
- **Files**: `0_core_understanding.ipynb`, `1_reward_engine.ipynb`, `orchestration.ipynb`, `orchestration_level_2.ipynb`, `reward_engine.py`, `grpo_loss.py`, `dataset.py`
- **Core Research Insights**: Replicated DeepSeek-R1-Zero's discovery that reasoning capabilities emerge purely through Reinforcement Learning without prior Supervised Fine-Tuning (SFT).

```
+-----------------------------------------------------------------------------------+
|                         GRPO RL Optimization Pipeline                             |
|                                                                                   |
|  Prompt (q) ---> Policy Model (π_θ) ---> Sample G Completions {o_1, o_2, ..., o_G} |
|                       |                                   |                       |
|                       v                                   v                       |
|               Reference Model (π_ref)             Rule-Based Reward Engine        |
|                       |                                (R_acc + R_fmt)            |
|                       v                                   |                       |
|               KL Divergence Penalty                       v                       |
|                       |                           Compute Group Advantage         |
|                       |                           A_i = (r_i - mean)/(std + ε)    |
|                       \                                   /                       |
|                        v                                 v                        |
|                         Clipped Surrogate Objective Loss                          |
+-----------------------------------------------------------------------------------+
```

- **GRPO Mathematical Objective**:
  $$\mathcal{J}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \left( \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) \right)$$
  where advantage $A_i = \frac{r_i - \text{mean}(\{r_1..r_G\})}{\text{std}(\{r_1..r_G\}) + \epsilon}$.
- **Unbiased Token-Level KL Divergence**:
  $$\mathbb{D}_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - 1$$
- **Rule-Based Reward Engine Implementation**:
  - Mathematical correctness reward $R_{\text{acc}}$ evaluated via SymPy symbolic verification.
  - Format tag reward $R_{\text{fmt}}$ enforcing `<think>...</think>` XML structure.
- **Multi-Stage Training Evolution**:
  - Cold-Start Warmup -> Annealed GRPO Loop -> Language Consistency Reward $R_{\text{lang}}$ (suppressing language mixing artifacts) -> Rejection Sampling & Distillation SFT Curation.

#### 3.2 MiniLLM: On-Policy Distillation of Large Language Models
- **Files**: `0-understanding.ipynb`, `implementation_.ipynb`
- **Core Research Insights**: Replicated MiniLLM's resolution of legacy distillation failures on generative LLMs. Standard Forward KL forces student models to over-estimate zero-probability void regions due to capacity mismatch. MiniLLM employs an On-Policy Reverse KL objective.

```
+-----------------------------------------------------------------------------------+
|                        MiniLLM Distillation Architecture                          |
|                                                                                   |
|  Prompt ---> Mixture Policy P_tilde = γ P_teacher + (1-γ) P_student               |
|                                     |                                             |
|                                     v                                             |
|                          Sample Trajectory y ~ P_tilde                            |
|                                     |                                             |
|                     +---------------+---------------+                             |
|                     |                               |                             |
|                     v                               v                             |
|          Teacher Logits P_teacher        Student Logits P_student                 |
|                     \                               /                             |
|                      v                             v                              |
|                 On-Policy Reverse KL Reward Calculation:                          |
|                 r_t = log P_teacher(y_t) - log P_student(y_t)                     |
|                                     |                                             |
|                                     v                                             |
|                Policy Gradient Loss Update + Single-Step Reg                      |
+-----------------------------------------------------------------------------------+
```

- **Reverse KL Formulation**:
  $$\mathbb{D}_{\text{KL}}(P_{\text{student}} || P_{\text{teacher}}) = \mathbb{E}_{y \sim P_{\text{student}}}\left[ \log P_{\text{student}}(y|x) - \log P_{\text{teacher}}(y|x) \right]$$
- **On-Policy Mixture Policy**:
  $$\tilde{P}(y|x) = \gamma P_{\text{teacher}}(y|x) + (1-\gamma) P_{\text{student}}(y|x)$$
- **Implementation**: Class `MiniLLMDistillationTrainer` in `implementation_.ipynb` implementing single-step token-level reverse KL rewards, teacher-mixed sampling, and student update loops.

---

## 3. Mathematical Reference & Performance Benchmark Matrix

### 3.1 Attention Architecture Comparison Matrix

| Attention Variant | Key-Value Cache Size per Token | VRAM Memory Scaling | Key Feature / Benefit |
| :--- | :--- | :--- | :--- |
| **Multi-Head Attention (MHA)** | $2 \cdot n_{\text{layers}} \cdot n_{\text{heads}} \cdot d_{\text{head}} \cdot b_{\text{bytes}}$ | High ($O(N)$ per request) | Baseline expressiveness; heavy memory bottleneck during generation. |
| **Grouped-Query Attention (GQA)** | $2 \cdot n_{\text{layers}} \cdot n_{\text{kv\_heads}} \cdot d_{\text{head}} \cdot b_{\text{bytes}}$ | Medium ($4\times\text{--}8\times$ reduction) | Standard in LLaMA 2/3; balances memory footprint and attention capacity. |
| **Multi-Head Latent Attention (MLA)** | $d_c^{KV} + d_R^{K}$ (Low-rank compressed state) | Minimal (Ultra-compressed) | DeepSeek-V2/V3/R1 mechanism; compresses KV states into latent vectors. |

### 3.2 Quantization & PEFT Tradeoff Matrix

| Technique | Precision / Data Format | Memory Footprint | Accuracy Retention | Primary Production Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Full Fine-Tuning** | FP16 / BF16 (16-bit) | $100\%$ ($16\times$ Model Size for AdamW) | $100\%$ (Baseline) | Pre-training, core base model domain adaptation. |
| **Standard LoRA** | FP16 / BF16 Base + Low-Rank $A, B$ | $< 5\%$ added VRAM | $98\%\text{--}99\%$ | Single-adapter task specialization. |
| **QLoRA** | NF4 Base + FP16 Low-Rank Adapters | $\sim 25\%\text{--}30\%$ of FP16 base | $97\%\text{--}99\%$ | Consumer GPU fine-tuning of large LLMs. |
| **DoRA** | NF4/FP16 Base + Weight-Decomposed LoRA | $< 6\%$ added VRAM | Matches Full FT | Fine-tuning tasks requiring nuanced directional adaptation. |
| **Int8 Dynamic KV-Cache** | Int8 Per-Token Quantized KV Buffers | $50\%$ reduction in KV VRAM | $> 99.5\%$ | High-concurrency autoregressive generation servers. |

### 3.3 Distillation Objective Comparison Matrix

| Distillation Method | Objective Function | Sampling Distribution | Behavior / Characteristics |
| :--- | :--- | :--- | :--- |
| **Forward KL** | $\mathbb{D}_{\text{KL}}(P_{\text{teacher}} \| Q_{\text{student}})$ | Teacher Off-Policy | **Mean-Seeking**: Student spreads probability across all teacher modes; causes hallucination on low-capacity models. |
| **Reverse KL (MiniLLM)** | $\mathbb{D}_{\text{KL}}(Q_{\text{student}} \| P_{\text{teacher}})$ | On-Policy Mixture ($\tilde{P}$) | **Mode-Seeking**: Student focuses strictly on major valid modes; eliminates low-probability void hallucinations. |
| **Subspace Distillation** | $\| W_{\text{proj}} h_{\text{student}} - h_{\text{teacher}} \|_2^2$ | Hidden Feature Layers | Direct intermediate representation matching across mismatched layer dimensions. |

---

## 4. Standalone Portfolio Project Profiles (GitHub / Portfolio Ready)

### Project Profile 1: Custom Modular LLM Engine & Architecture Playground
* **Repository Domain**: `Build_From_Scratch/`
* **Description**: A ground-up PyTorch implementation of decoder-only transformer LLMs, reproducing modern architectural advancements including RoPE positional embeddings, RMSNorm, SwiGLU FFNs, Grouped-Query Attention (GQA), and DeepSeek Multi-Head Latent Attention (MLA). Includes dynamic quantization engines (Int8 KV Cache, FP8, MXFP4) and PEFT modules (LoRA, QLoRA, DoRA).
* **Key Achievements**: Implemented DeepSeek MLA latent KV-compression reducing memory bandwidth demands; built custom Muon matrix-orthogonalizing optimizer using Newton-Schulz iterations.

### Project Profile 2: High-Throughput Async LLM Inference Server & Profiling Harness
* **Repository Domain**: `Inference-and-model-serving/`
* **Description**: An enterprise-grade asynchronous model serving infrastructure built with FastAPI, supporting token-by-token Server-Sent Events (SSE) streaming, `torch.compile` integration, and CUDA Graph execution with dynamic bucket padding. Includes an automated 4D benchmarking suite and profiling integration with PyTorch Profiler and NVIDIA Nsight Systems (`nsys`).
* **Key Achievements**: Eliminated GPU kernel launch overhead via bucket-padded CUDA Graphs; engineered real-time metrics tracking for TTFT, ITL, output throughput, and latency percentiles.

### Project Profile 3: DeepSeek-R1 & MiniLLM Alignment & Distillation Suite
* **Repository Domain**: `research-paper-imlementations/`
* **Description**: End-to-end open-source replications of cutting-edge LLM alignment and distillation papers:
  1. **DeepSeek-R1 & DeepSeek-R1-Zero**: Critic-free Group Relative Policy Optimization (GRPO) reinforcement learning engine paired with SymPy-based rule rewards, driving the emergence of reasoning chains (`<think>`) and language consistency.
  2. **MiniLLM Distillation**: On-policy reverse-KL distillation trainer overcoming forward-KL mode-spreading artifacts in small student models.
* **Key Achievements**: Formulated group-normalized advantage estimators and unbiased token-level KL penalties; verified reasoning trajectory expansion during GRPO training loops.

---
*Document automatically generated from code inspection and notebook analysis across all workspace repositories.*
