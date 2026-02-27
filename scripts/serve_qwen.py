"""Lightweight OpenAI-compatible API server for Qwen3-8B using transformers.

Usage:
  CUDA_VISIBLE_DEVICES=6 python scripts/serve_qwen.py --port 8234
"""

import argparse
import json
import time
import uuid
import threading
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from transformers import AutoModelForCausalLM, AutoTokenizer

model = None
tokenizer = None
_inference_lock = threading.Lock()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def load_model(model_name, dtype):
    global model, tokenizer
    print(f"Loading {model_name} (dtype={dtype})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Tokenizer loaded. Loading model weights to GPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print("Weights loaded in CPU. Moving to GPU...", flush=True)
    model = model.to("cuda")
    model.eval()
    torch.cuda.empty_cache()
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"Model on {next(model.parameters()).device} | GPU mem: {mem_gb:.1f} GB", flush=True)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if "/chat/completions" in self.path:
            self._handle_chat()
        elif "/completions" in self.path:
            self._handle_chat()
        else:
            self._send(404, {"error": "not found"})

    def do_GET(self):
        if "/models" in self.path:
            self._send(200, {
                "data": [{"id": model.config._name_or_path, "object": "model"}]
            })
        elif "/health" in self.path:
            self._send(200, {"status": "ok"})
        else:
            self._send(200, {"status": "ok"})

    def _handle_chat(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])
        pass  # messages ready
        max_tokens = body.get("max_tokens", 1024)
        temperature = body.get("temperature", 0.7)

        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        internal_max = max(max_tokens * 4, 2048)

        t0 = time.time()
        with _inference_lock, torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=internal_max,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0.01,
                top_p=0.95,
            )
        gen_ids = outputs[0][prompt_len:]
        content = tokenizer.decode(gen_ids, skip_special_tokens=True)
        elapsed = time.time() - t0

        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if content.startswith("<think>"):
            content = content.split("</think>")[-1].strip() if "</think>" in content else ""

        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "qwen3-8b"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": len(gen_ids),
                "total_tokens": prompt_len + len(gen_ids),
            },
        }
        self._send(200, resp)
        tps = len(gen_ids) / max(elapsed, 0.01)
        print(f"  [{tps:.1f} tok/s] {len(gen_ids)} tokens in {elapsed:.1f}s")

    def _send(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--port", type=int, default=8234)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    load_model(args.model, args.dtype)
    server = ThreadedHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Serving {args.model} on http://0.0.0.0:{args.port}/v1", flush=True)
    print(f"  POST /v1/chat/completions  |  GET /v1/models", flush=True)
    print(f"  Multi-threaded with inference lock", flush=True)
    server.serve_forever()
