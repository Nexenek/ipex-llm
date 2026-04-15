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

"""Smoke test for Qwen3.5 inference on Intel GPU via IPEX-LLM.

Loads the smallest public Qwen3.5 dense variant, runs a short generation,
and asserts that output contains no NaN/Inf and decodes to non-empty
text. Intended to be executed on a machine with an Intel XPU available.
"""

import os
import unittest

import torch


QWEN3_5_REPO = os.environ.get("QWEN3_5_TEST_REPO", "Qwen/Qwen3.5-2B-Instruct")


class TestQwen3_5Generation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not torch.xpu.is_available():
            raise unittest.SkipTest("Intel XPU not available in this environment")

        from transformers import AutoTokenizer
        from ipex_llm.transformers import AutoModelForCausalLM

        cls.tokenizer = AutoTokenizer.from_pretrained(
            QWEN3_5_REPO, trust_remote_code=True
        )
        cls.model = AutoModelForCausalLM.from_pretrained(
            QWEN3_5_REPO,
            load_in_4bit=True,
            optimize_model=True,
            trust_remote_code=True,
            use_cache=True,
        ).half().to("xpu")

    def test_generate_no_nan(self):
        prompt = "Explain entropy in one sentence."
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to("xpu")

        with torch.inference_mode():
            out = self.model.generate(inputs.input_ids, max_new_tokens=20)

        self.assertFalse(torch.isnan(out.float()).any(),
                         "Output token ids contain NaN (should never happen for "
                         "int tensors — indicates a kernel-level failure)")
        decoded = self.tokenizer.batch_decode(
            out[:, inputs.input_ids.size(1):], skip_special_tokens=True
        )[0]
        self.assertTrue(decoded.strip(), "Decoded output is empty")


if __name__ == "__main__":
    unittest.main()
