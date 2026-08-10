"""
02_train_distill.py - V2 (AMD MI300X) Sequence-Level Knowledge Distillation
Student: openbmb/MiniCPM5-1B | Teacher outputs generated offline in 01_generate_teacher.py
Dataset: 15 Aligned Multi-Domain Pairs (data/teacher_dataset.json)
Params: CE-only (no KL, no teacher loaded at train time), LR=2e-5, Weight Decay=0.01, epoch count set by NUM_EPOCHS
Hardware: AMD Instinct MI300X (192GB) via SSH, device-agnostic (CUDA/ROCm/CPU),
          GPU credits provided by the AMD AI Developer Program
Output Checkpoint: ./models/distilled_minicpm5_1b_v2_amd
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup


def resolve_device_and_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        return device, torch.float32
    dtype = torch.bfloat16
    try:
        if not torch.cuda.is_bf16_supported():
            dtype = torch.float16
        else:
            torch.ones(1, device=device, dtype=torch.bfloat16).sum()
    except Exception:
        dtype = torch.float16
    return device, dtype


class JSONDataset(Dataset):
    def __init__(self, json_path, student_tokenizer, max_len=1024):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.student_tok = student_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        target = item["teacher_json"]

        p_ids = self.student_tok.encode(prompt, add_special_tokens=True)
        t_ids = self.student_tok.encode(target, add_special_tokens=False) + [self.student_tok.eos_token_id]

        full_s_ids = (p_ids + t_ids)[:self.max_len]
        s_labels = [-100] * len(p_ids) + t_ids[:self.max_len - len(p_ids)]
        s_labels = s_labels[:len(full_s_ids)]

        if all(label == -100 for label in s_labels):
            raise ValueError(
                f"Example {idx} produced an all-masked label sequence: the prompt alone "
                f"({len(p_ids)} tokens) already exceeds max_len={self.max_len}. "
                "Truncate the prompt or increase max_len."
            )

        pad_len = self.max_len - len(full_s_ids)
        s_input_ids = full_s_ids + [self.student_tok.pad_token_id] * pad_len
        s_mask = [1] * len(full_s_ids) + [0] * pad_len
        labels = s_labels + [-100] * pad_len

        return {
            "s_input_ids": torch.tensor(s_input_ids, dtype=torch.long),
            "s_mask": torch.tensor(s_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def main():
    STUDENT_ID = "openbmb/MiniCPM5-1B"
    DATA_PATH = "./data/teacher_dataset.json"
    OUTPUT_DIR = "./models/distilled_minicpm5_1b_v2_amd"

    NUM_EPOCHS = 2

    SEED = 42
    torch.manual_seed(SEED)

    device, dtype = resolve_device_and_dtype()
    print(f"[*] Resolved device={device}, dtype={dtype}")

    print(f"[*] Loading Student Tokenizer...")
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID, trust_remote_code=True)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    print(f"[*] Loading Student Model ({STUDENT_ID})...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID,
        dtype=dtype,
        trust_remote_code=True
    ).to(device)

    dataset = JSONDataset(DATA_PATH, student_tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(student.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=len(dataloader) * NUM_EPOCHS)

    print(f"[*] Starting V2 (AMD MI300X) Sequence-Level KD Loop (15 Multi-Domain Pairs, {NUM_EPOCHS} Epochs)...")
    student.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            s_ids = batch["s_input_ids"].to(device)
            s_mask = batch["s_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            s_out = student(input_ids=s_ids, attention_mask=s_mask)

            shift_logits = s_out.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            print(f"V2 (AMD MI300X) | Epoch {epoch+1} | Step {step+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

        print(f"=== V2 (AMD MI300X) Epoch {epoch+1} Avg Loss: {epoch_loss / len(dataloader):.4f} ===")

    print(f"[+] Saving V2 (AMD MI300X) checkpoint to {OUTPUT_DIR}...")
    student.save_pretrained(OUTPUT_DIR)
    student_tokenizer.save_pretrained(OUTPUT_DIR)
    print("[+] V2 (AMD MI300X) distillation training completed successfully!")


if __name__ == "__main__":
    main()
