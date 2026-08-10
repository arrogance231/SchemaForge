"""
01_generate_teacher.py - Iteration 3: Multi-Domain Prompt Alignment & System Persona
Generates teacher targets from the hard-example training corpus using a
standardized system prompt + chat format, gating every raw output through the
schemaforge §4.2 validation gate before admitting it to the training dataset.
Teacher: google/gemma-4-31B -> Student: openbmb/MiniCPM5-1B

Corpus: ./data/hard_examples_train.jsonl (written by
``python -m schemaforge.hardexamples.generate``), one JSON object per line with
keys ``schema``, ``source_text``, ``reference``, ``tags``.

Pipeline per record: build a prompt from the schema name + its semantic fields,
run generation (batched vLLM when a working build is importable, otherwise the
batched HuggingFace ``AutoModelForCausalLM.generate`` fallback), extract clean
JSON with the balanced-brace scanner, then run
``schemaforge.validation.gate.validate_teacher_output``. Accepted outputs are
appended to the admitted training set; rejected ones land in a separate
rejections file for inspection. Source-support checking runs with
``fuzzy_support=True``, deliberately crediting typo/OCR-denoising near-matches
instead of literal substring/ontology matches only.

Two code paths: batched vLLM sampling when a working vLLM build is importable,
otherwise a batched HuggingFace ``AutoModelForCausalLM.generate`` fallback.
Generation is greedy (``temperature=0.0``, no sampling) in both paths: this is a
labeling teacher whose output must be reproducible across runs -- nonzero
temperature would reintroduce run-to-run variance in the ground truth this
project's comparisons depend on.
Teacher generation runs on an AMD Instinct MI300X server via SSH (ROCm); no
ROCm vLLM build is present there, so the HF `generate` fallback is what executes.
"""

import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemaforge import get_schema  # noqa: E402
from schemaforge.validation import gate  # noqa: E402

TRAIN_CORPUS_FILE = "./data/hard_examples_train.jsonl"
OUTPUT_FILE = "./data/teacher_dataset.json"
REJECTIONS_FILE = "./data/teacher_dataset_rejections.json"
MODEL_NAME = "google/gemma-4-31B"


def VLLM_AVAILABLE() -> bool:
    """Probe whether a working vLLM install exists without importing it eagerly.

    Returns ``True`` when the ``vllm`` module is findable.  The import itself
    is deferred to :func:`main` so that this file imports cleanly on hosts
    without a working vLLM build -- the CPU-only development machine, or an
    AMD/ROCm server where vLLM ROCm is not installed (research direction §9).
    """
    return importlib.util.find_spec("vllm") is not None


def build_prompt(schema_name: str, document_text: str) -> str:
    """Build the teacher prompt for one corpus record.

    Includes the schema name and the sorted list of ``semantic_fields`` (the
    fields the model is actually asked to extract -- deterministic_fields are
    out of scope per research direction §0).  Deterministic and free of any
    randomness/timestamps.
    """
    spec = get_schema(schema_name)
    fields = ", ".join(sorted(spec.semantic_fields))
    return (
        f"Extract the following fields as JSON from the text below. "
        f"Schema: {schema_name}.\n"
        f"Fields to extract: {fields}.\n"
        f"Text:\n{document_text}\n"
        f"JSON Output:"
    )


def extract_json_str(text: str) -> str:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start = text.find("{")
    if start == -1:
        return text.strip()

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1].strip()

    return text[start:].strip()


def generate_with_hf(model_name, prompts, max_new_tokens=512, temperature=0.0):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    batch_size = 4
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    generated_texts = []

    for batch_idx, start in enumerate(range(0, len(prompts), batch_size), start=1):
        batch = prompts[start:start + batch_size]
        print(f"[*] HF generate fallback: batch {batch_idx}/{n_batches}")
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        input_seq_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id
            )
        new_tokens = generated[:, input_seq_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        generated_texts.extend(text.strip() for text in decoded)

    return generated_texts


def load_train_corpus(corpus_file: str) -> list[dict]:
    """Load the hard-example training corpus as a list of plain dicts.

    One JSON object per line with keys ``schema``, ``source_text``, ``reference``,
    ``tags`` (the format ``schemaforge.hardexamples.generate.serialize_records``
    writes).  Raises ``FileNotFoundError`` if the corpus has not been generated.
    """
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(
            f"training corpus not found: {corpus_file!r}. "
            f"Run `python -m schemaforge.hardexamples.generate` first to build it."
        )
    records = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    train_records = load_train_corpus(TRAIN_CORPUS_FILE)
    print(f"[*] Loaded {len(train_records)} training records from {TRAIN_CORPUS_FILE}")

    prompts = [build_prompt(record["schema"], record["source_text"]) for record in train_records]

    if VLLM_AVAILABLE():
        from vllm import LLM, SamplingParams

        print(f"[*] Initializing vLLM for Iteration 3 31B Teacher: {MODEL_NAME}...")
        llm = LLM(
            model=MODEL_NAME,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=True,
            trust_remote_code=True
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512
        )

        print(f"[*] Running Iteration 3 teacher inference on {len(prompts)} prompts...")
        outputs = llm.generate(prompts, sampling_params)

        raw_outputs = []
        for output in outputs:
            raw_outputs.append(output.outputs[0].text.strip())
    else:
        print(
            f"[*] vLLM not available; using batched HuggingFace `generate` "
            f"fallback for {MODEL_NAME}..."
        )
        raw_outputs = generate_with_hf(
            MODEL_NAME,
            prompts,
            max_new_tokens=512,
            temperature=0.0
        )

    records = []
    rejections = []
    gate_results = []
    for record, prompt, raw_text in zip(train_records, prompts, raw_outputs):
        clean_json_text = extract_json_str(raw_text)
        result = gate.validate_teacher_output(
            record["schema"], record["source_text"], clean_json_text, fuzzy_support=True
        )
        gate_results.append(result)
        if result.accepted:
            records.append({
                "prompt": prompt,
                "document_text": record["source_text"],
                "schema": record["schema"],
                "tags": record["tags"],
                "teacher_json": json.dumps(result.parsed, sort_keys=True)
            })
        else:
            rejections.append({
                "schema": record["schema"],
                "source_text": record["source_text"],
                "tags": record["tags"],
                "raw_teacher_output": raw_text,
                "reasons": result.reasons
            })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    with open(REJECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(rejections, f, indent=2)

    if gate_results:
        rejection_rate = gate.rejection_rate(gate_results)
    else:
        rejection_rate = 0.0
        print("[+] Gate: no records processed")

    print(
        f"[+] Gate: {len(records)}/{len(gate_results)} admitted "
        f"({rejection_rate:.1%} rejected). "
        f"See {REJECTIONS_FILE} for reasons."
    )
    print(f"[+] Saved {len(records)} admitted teacher training pairs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
