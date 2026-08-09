import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

STUDENT_ID = "openbmb/MiniCPM-1B-sft-bf16"
DATA_PATH = "./data/teacher_dataset.json"

tok = AutoTokenizer.from_pretrained(STUDENT_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

cfg = AutoConfig.from_pretrained(STUDENT_ID, trust_remote_code=True)
if hasattr(cfg, "rope_scaling") and isinstance(cfg.rope_scaling, dict):
    cfg.rope_scaling["type"] = "linear"
    if "factor" not in cfg.rope_scaling:
        cfg.rope_scaling["factor"] = 1.0
else:
    cfg.rope_scaling = {"type": "linear", "factor": 1.0}

model = AutoModelForCausalLM.from_pretrained(
    STUDENT_ID,
    config=cfg,
    torch_dtype=torch.float32,
    trust_remote_code=True
).cuda()

with open(DATA_PATH, "r") as f:
    data = json.load(f)

for i, item in enumerate(data):
    full_text = item["prompt"] + item["teacher_json"]
    inputs = tok(full_text, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    logits = outputs.logits
    print(f"Sample {i}: Logits NaN = {torch.isnan(logits).any().item()}, shape = {logits.shape}")
