"""
04_eval_public_dataset.py - Iteration 3 Final Benchmark Evaluation
Evaluates ./models/distilled_minicpm5_1b_iter3 on suneeldk/text-json public dataset
"""

import time
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def benchmark_public_dataset(model_path, dataset_id="suneeldk/text-json", num_samples=10):
    print(f"\n==================================================")
    print(f"[*] Benchmarking Public Dataset: {dataset_id}")
    print(f"[*] Target Checkpoint : {model_path}")
    print(f"==================================================")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    ds = load_dataset(dataset_id, split="train")

    valid_json_count = 0
    total_tokens = 0
    total_elapsed = 0.0

    print(f"\n[*] Running Benchmark Evaluation on {min(num_samples, len(ds))} Samples...\n")

    for idx in range(min(num_samples, len(ds))):
        row = ds[idx]
        doc_text = str(row.get("text") or row.get("input") or str(row))[:500]

        prompt = f"Extract structured JSON from the text:\n{doc_text}\nJSON Output:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        end = time.perf_counter()

        elapsed = end - start
        gen_tokens = outputs[0].size(0) - inputs["input_ids"].size(1)

        total_elapsed += elapsed
        total_tokens += gen_tokens

        res_text = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)

        json_str = res_text.strip()
        if "{" in json_str:
            s_idx = json_str.find("{")
            e_idx = json_str.find("}", s_idx)
            if e_idx != -1:
                json_str = json_str[s_idx:e_idx + 1]

        try:
            json.loads(json_str)
            is_valid = True
            valid_json_count += 1
        except Exception:
            is_valid = False

        print(f"Sample {idx+1}/{min(num_samples, len(ds))} | Valid JSON: {is_valid} | Time: {elapsed:.2f}s | Tokens: {gen_tokens}")

    avg_tps = total_tokens / total_elapsed if total_elapsed > 0 else 0
    accuracy = (valid_json_count / min(num_samples, len(ds))) * 100

    print("\n" + "=" * 60)
    print(f" ITERATION 3 FINAL PUBLIC DATASET BENCHMARK RESULTS ({dataset_id})")
    print("=" * 60)
    print(f" Total Samples Evaluated : {min(num_samples, len(ds))}")
    print(f" Valid JSON Outputs      : {valid_json_count} / {min(num_samples, len(ds))}")
    print(f" JSON Accuracy           : {accuracy:.1f}%")
    print(f" Total Time              : {total_elapsed:.2f} seconds")
    print(f" Average Throughput      : {avg_tps:.2f} tokens/sec")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    benchmark_public_dataset("./models/distilled_minicpm5_1b_iter3", dataset_id="suneeldk/text-json", num_samples=10)
