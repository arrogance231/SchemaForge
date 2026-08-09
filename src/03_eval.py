"""
03_eval.py
Runs empirical benchmarks comparing baseline MiniCPM5-1B and distilled checkpoint for throughput, latency, and JSON structure validity.
"""

import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def eval_model(model_name_or_path, label):
    print(f"\n==================================================")
    print(f"[*] Benchmark Running: {label} ({model_name_or_path})")
    print(f"==================================================")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    test_prompt = "Extract entity details into JSON with keys 'invoice_number', 'vendor', 'date', 'amount':\nInvoice INV-99281, Acme Enterprise Solutions Inc, Date 2026-08-03, Total Amount $12,450.00.\nJSON Output:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    end = time.perf_counter()

    tokens_gen = out[0].size(0) - inputs["input_ids"].size(1)
    elapsed = end - start
    tps = tokens_gen / elapsed if elapsed > 0 else 0

    res_text = tokenizer.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)

    # Extract first JSON block up to first closing brace
    json_str = res_text.strip()
    if "{" in json_str:
        start_idx = json_str.find("{")
        end_idx = json_str.find("}", start_idx)
        if end_idx != -1:
            json_str = json_str[start_idx:end_idx + 1]

    # Validate JSON syntax
    try:
        parsed_json = json.loads(json_str)
        json_valid = True
    except Exception:
        json_valid = False

    print(f"Generated Output:\n{json_str}")
    print("-" * 50)
    print(f"JSON Syntax Valid : {json_valid}")
    print(f"Generation Time   : {elapsed:.3f} seconds")
    print(f"Tokens Generated  : {tokens_gen} tokens")
    print(f"Throughput        : {tps:.2f} tokens/sec")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    print("[*] Starting Retraining Comparison Benchmark Suite...")
    eval_model("./models/distilled_exp1_multitask", "Exp 1: Multi-Task Distillation (Alpha=0.5, Temp=2.0, LR=3e-5)")
    eval_model("./models/distilled_exp2_softkl", "Exp 2: Soft-Weighted KL Dominance (Alpha=0.2, Temp=1.5, LR=5e-5)")
