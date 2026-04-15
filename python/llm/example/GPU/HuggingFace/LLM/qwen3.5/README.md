# Qwen3.5

In this directory, you will find examples on how you could apply IPEX-LLM INT4
optimizations on Qwen3.5 models on [Intel GPUs](../../../README.md). For
illustration purposes, we utilize
[Qwen/Qwen3.5-2B-Instruct](https://huggingface.co/Qwen/Qwen3.5-2B-Instruct) as
a reference Qwen3.5 model.

Qwen3.5 introduces a **hybrid** architecture: Gated DeltaNet (linear-attention)
layers are interleaved with standard Gated-Attention layers. IPEX-LLM's
Qwen3.5 optimization pass fuses the QKV projections, installs the IPEX SDPA
kernel on the full-attention layers, and leaves the DeltaNet blocks on
the HuggingFace reference forward in this first landing — correctness first,
a fused SYCL/IPEX DeltaNet kernel is a follow-up.

## 0. Requirements

To run these examples with IPEX-LLM on Intel GPUs, we have some recommended
requirements for your machine, please refer to
[here](../../../README.md#requirements) for more information.

Qwen3.5 support also requires `transformers` recent enough to ship the
`Qwen3_5` / `Qwen3Next` modelling classes.

## Example: Predict Tokens using `generate()` API

In the example [generate.py](./generate.py), we show a basic use case for a
Qwen3.5 model to predict the next N tokens using `generate()` API, with
IPEX-LLM INT4 optimizations on Intel GPUs.

### 1. Install

Follow the instructions in the parent README.

### 2. Configures OneAPI environment variables for Linux

```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Run

```bash
python ./generate.py --repo-id-or-model-path Qwen/Qwen3.5-2B-Instruct \
    --prompt "AI是什么？" --n-predict 64
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: huggingface repo id or
  local path for the Qwen3.5 model. Defaults to
  `'Qwen/Qwen3.5-2B-Instruct'`.
- `--prompt PROMPT`: argument defining the prompt to be infered.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to
  predict. Defaults to `32`.
- `--modelscope`: use ModelScope as the model hub instead of HuggingFace.

#### Sample Output

```
Inference time: 3.14 s
-------------------- Prompt --------------------
AI是什么？
-------------------- Output --------------------
AI是人工智能的缩写，是计算机科学的一个分支……
```
