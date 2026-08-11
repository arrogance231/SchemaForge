"""V3 iteration 2: rebuild the training dataset with the fuzzy-support gate enabled.

Recovers admissions from V3 iter 1's saved rejections by re-running the gate with
fuzzy_support=True. Teacher generation is deterministic (greedy, temp=0.0), so the
saved raw outputs are byte-identical to what a fresh iter-2 run would produce;
this recovers the fuzzy-gate admissions without re-running generation (~534/1443
step-3 recoveries measured in the validation dry-run).
"""

import importlib.util
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemaforge.validation import gate  # noqa: E402


SPEC = importlib.util.spec_from_file_location(
    "gen_teacher", os.path.join(os.path.dirname(__file__), "01_generate_teacher.py")
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MOD)
build_prompt = MOD.build_prompt

ADMITTED = "./data/teacher_dataset_v3iter1.json"
REJECTIONS = "./data/teacher_dataset_rejections_v3iter1_bak.json"
OUT = "./data/teacher_dataset_v3iter2.json"


def main() -> None:
    with open(ADMITTED, "r", encoding="utf-8") as f:
        admitted = json.load(f)
    with open(REJECTIONS, "r", encoding="utf-8") as f:
        rejected = json.load(f)

    print(f"[+] admitted: {len(admitted)}")
    print(f"[+] rejected: {len(rejected)}")
    if admitted:
        print(f"[+] admitted[0] keys: {sorted(admitted[0].keys())}")
    if rejected:
        print(f"[+] rejected[0] keys: {sorted(rejected[0].keys())}")
        print(f"[+] rejected[0] reasons: {rejected[0].get('reasons')}")

    recovered = []
    still_rejected = Counter()
    for rec in rejected:
        result = gate.validate_teacher_output(
            rec["schema"], rec["source_text"], rec["raw_teacher_output"], fuzzy_support=True
        )
        if result.accepted:
            recovered.append({
                "prompt": build_prompt(rec["schema"], rec["source_text"]),
                "document_text": rec["source_text"],
                "schema": rec["schema"],
                "tags": rec["tags"],
                "teacher_json": json.dumps(result.parsed, sort_keys=True),
            })
        else:
            still_rejected[" / ".join(result.reasons)] += 1

    merged = admitted + recovered
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    total = len(admitted) + len(rejected)
    new_rate = len(merged) / total
    print(f"[+] recovered: {len(recovered)}/{len(rejected)} ({len(recovered)/max(len(rejected),1):.1%})")
    print(f"[+] merged: {len(merged)}/{total} admitted ({(1-new_rate):.1%} rejected)")
    print(f"[+] still-rejected by reason (top 10):")
    for reason, count in still_rejected.most_common(10):
        print(f"    {count:5d}  {reason}")
    print(f"[+] spot-check of recovered examples:")
    for rec in recovered[:6]:
        print(f"    schema={rec['schema']} | src={rec['document_text'][:60]!r} | out={rec['teacher_json'][:80]}")
    print(f"[+] wrote {OUT} ({len(merged)} records)")


if __name__ == "__main__":
    main()
