# Serving Qwen3.5 GGUFs with IPEX-LLM

This example loads a **Qwen3.5** GGUF file on Intel GPU via IPEX-LLM and
exposes it behind an OpenAI-compatible FastAPI endpoint.

## 1. Install

Follow the base Lightweight-Serving install instructions in
[`README.md`](./README.md), then make sure `transformers` is recent enough
to ship the `Qwen3_5` / `Qwen3Next` modelling classes.

```bash
pip install -e python/llm
```

## 2. Get a Qwen3.5 GGUF

Any GGUF produced by a `llama.cpp` version that includes the Qwen3.5 arch
support (upstream PR #19468) will work. Examples:

- Dense: `unsloth/Qwen3.5-2B-Instruct-GGUF` → `Qwen3.5-2B-Instruct.Q4_0.gguf`
- Dense: `unsloth/Qwen3.5-9B-Instruct-GGUF`
- MoE:   `unsloth/Qwen3.5-35B-A3B-Instruct-GGUF`

Keep the matching `tokenizer.json` / `tokenizer_config.json` next to the
`.gguf` file — the loader picks them up automatically and falls back to a
GPT-2-style BPE rebuild from the GGUF metadata only if they are absent.

## 3. Serve

```bash
source /opt/intel/oneapi/setvars.sh
python qwen3_5_gguf_serving.py \
    --model-path /models/Qwen3.5-9B-Instruct.Q4_0.gguf \
    --low-bit sym_int4 \
    --port 8000
```

## 4. Query

```bash
curl http://localhost:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{
      "model": "qwen3.5",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain entropy in one sentence."}
      ],
      "max_tokens": 128
    }'
```

The server is fully OpenAI-API-compatible, so drop-in clients such as the
official `openai` Python SDK, `litellm`, or `open-webui` work by pointing
their base URL at `http://<host>:8000/v1`.

## Notes

- The hybrid architecture (Gated DeltaNet + standard attention) is
  dispatched automatically based on the GGUF's `full_attention_interval`
  metadata. Full-attention layers run through the IPEX SDPA kernel;
  DeltaNet layers currently run through HuggingFace's reference forward.
- For MoE GGUFs (`qwen35moe`) on limited-VRAM GPUs, set
  `IPEX_LLM_MOE_EXPERTS_ON_CPU=1` before launching to offload expert
  weights to CPU memory.
