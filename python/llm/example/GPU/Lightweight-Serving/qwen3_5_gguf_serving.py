#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# End-to-end example: load a Qwen3.5 GGUF file on Intel GPU via IPEX-LLM
# and serve it behind an OpenAI-compatible FastAPI endpoint.
#
# Usage:
#   python qwen3_5_gguf_serving.py \
#       --model-path /models/Qwen3.5-9B-Instruct.Q4_0.gguf \
#       --port 8000
#
# Then:
#   curl http://localhost:8000/v1/chat/completions \
#       -H 'content-type: application/json' \
#       -d '{"model":"qwen3.5","messages":[{"role":"user","content":"hi"}]}'

import argparse
import asyncio
import os
import uvicorn

from transformers import AutoTokenizer

from ipex_llm.serving.fastapi import FastApp, ModelWorker


async def main():
    parser = argparse.ArgumentParser(
        description='Serve a Qwen3.5 GGUF model on Intel GPU with ipex-llm')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to a Qwen3.5 .gguf file, or the HuggingFace '
                             'repo id of a Qwen3.5 model.')
    parser.add_argument('--low-bit', type=str, default='sym_int4',
                        help='Low-bit quantization type (sym_int4, sym_int8, ...).')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to bind the FastAPI server on.')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host interface to bind on.')

    args = parser.parse_args()

    local_model = ModelWorker(args.model_path, args.low_bit)

    # Prefer the tokenizer that the GGUF loader already reconstructed;
    # if we're pointed at a HuggingFace repo id or local directory
    # instead of a ``.gguf`` file, fall back to AutoTokenizer.
    tokenizer = getattr(local_model, "tokenizer", None)
    if tokenizer is None:
        tokenizer_source = args.model_path
        if tokenizer_source.endswith(".gguf"):
            tokenizer_source = os.path.dirname(os.path.abspath(tokenizer_source))
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    myapp = FastApp(local_model, tokenizer)
    config = uvicorn.Config(app=myapp.app, host=args.host, port=args.port)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
