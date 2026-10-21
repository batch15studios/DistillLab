"""
prune_dataset.py - Semantic De-Duplication and Diversity Pruning for DistillLab
Eliminates near-duplicate prompt concepts and redundant code implementations.
"""

import json
import re
import os
from pathlib import Path
from app.distill_engine import validate_sample, sanitize_text, estimate_tokens, SFTEntry

DATASET_LOCAL = "data/cs_programming_dataset.jsonl"
DATASET_DOWNLOADS = r"C:\Users\Ryan\Downloads\batchcode_latest\datacs_programming_dataset.jsonl"

def extract_features(item):
    inst = ""
    resp = ""
    sys_prompt = ""
    for m in item.get("messages", []):
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            inst = content
        elif role == "assistant":
            resp = content
        elif role == "system":
            sys_prompt = content
            
    code = " ".join(re.findall(r"```(?:[a-zA-Z0-9\+\#]*)\n([\s\S]*?)```", resp))
    inst_words = set(re.findall(r"[a-zA-Z]{3,}", inst.lower())) - {
        "the", "and", "for", "with", "implement", "write", "create", "complete",
        "production", "grade", "senior", "level", "using", "robust", "code", "function",
        "program", "demonstrate", "detailed", "solution", "please", "following", "task"
    }
    code_tokens = set(re.findall(r"[a-zA-Z_]{3,}", code.lower()))
    return inst, inst_words, code_tokens, len(resp), sys_prompt, resp

def main():
    if not os.path.exists(DATASET_LOCAL):
        print(f"Error: {DATASET_LOCAL} does not exist.")
        return

    with open(DATASET_LOCAL, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"============================================================")
    print(f"  DistillLab -> Semantic De-Duplication & Quality Pruner")
    print(f"  Input Dataset:    {len(items)} samples")
    print(f"============================================================")

    features = [extract_features(item) for item in items]
    kept_indices = []
    pruned_records = []

    for idx, (inst, i_words, c_tokens, r_len, sys_prompt, resp) in enumerate(features):
        duplicate = False
        match_info = None

        for kept_idx in kept_indices:
            k_inst, k_i_words, k_c_tokens, k_r_len, _, _ = features[kept_idx]
            
            inst_sim = len(i_words & k_i_words) / len(i_words | k_i_words) if (i_words and k_i_words) else 0.0
            code_sim = len(c_tokens & k_c_tokens) / len(c_tokens | k_c_tokens) if (c_tokens and k_c_tokens) else 0.0

            # Prune if instruction concept overlap >= 40% OR code structure overlap >= 65% OR strong hybrid overlap
            if inst_sim >= 0.40 or code_sim >= 0.65 or (inst_sim >= 0.30 and code_sim >= 0.45):
                duplicate = True
                match_info = (idx, kept_idx, inst_sim, code_sim, inst[:80])
                break

        if duplicate:
            pruned_records.append(match_info)
        else:
            # Re-sanitize and verify
            sanitized_messages = []
            for m in items[idx].get("messages", []):
                sanitized_messages.append({
                    "role": m.get("role", ""),
                    "content": sanitize_text(m.get("content", ""))
                })
            items[idx]["messages"] = sanitized_messages
            kept_indices.append(idx)

    pruned_items = [items[i] for i in kept_indices]

    print(f"\n[Deduplication Results]")
    print(f"  Original Count:       {len(items)}")
    print(f"  Redundancies Pruned:  {len(pruned_records)} samples")
    print(f"  Final High-Diversity: {len(pruned_items)} samples")

    print("\n[Examples of Eliminated Redundancies]")
    for p in pruned_records[:8]:
        print(f"  - Pruned #{p[0]+1} (concept overlap {p[2]:.2f} with #{p[1]+1}): {p[4]}...")

    # Write back clean dataset to local and downloads
    with open(DATASET_LOCAL, "w", encoding="utf-8") as f:
        for item in pruned_items:
            f.write(json.dumps({"messages": item["messages"]}, ensure_ascii=False) + "\n")

    if os.path.exists(os.path.dirname(DATASET_DOWNLOADS)):
        with open(DATASET_DOWNLOADS, "w", encoding="utf-8") as f:
            for item in pruned_items:
                f.write(json.dumps({"messages": item["messages"]}, ensure_ascii=False) + "\n")
        print(f"\nSuccessfully synced clean dataset to {DATASET_DOWNLOADS}")

    print(f"Successfully pruned local dataset at {DATASET_LOCAL}")
    print("============================================================\n")

if __name__ == "__main__":
    main()
