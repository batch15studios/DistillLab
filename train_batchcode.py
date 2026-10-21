"""
train_batchcode.py - Headless Unsloth Fine-Tuning Pipeline for 'batchcode'
Target Architecture: Qwen/Qwen2.5-Coder-1.5B-Instruct
Quantization: 4-bit base (bitsandbytes) + Q4_K_M GGUF export for Ollama
"""

import os
import sys
import argparse
import json
from pathlib import Path

def sanitize_dataset(file_path: str) -> int:
    """Pre-flight check to strip markdown headers and invalid C/C++ preprocessor directives."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()
    
    count_template = content.count("#template")
    count_requires = content.count("#requires")
    count_concept = content.count("#concept")
    total_anomalies = count_template + count_requires + count_concept
    
    if total_anomalies > 0:
        print(f"[Sanitizer] Cleaning {total_anomalies} preprocessor anomalies in {file_path}...")
        cleaned = content.replace("#template", "template").replace("#requires", "requires").replace("#concept", "concept")
        with open(p, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print("[Sanitizer] Sanitization complete: #template, #requires, and #concept sanitized.")
    else:
        print(f"[Sanitizer] Dataset {file_path} is 100% clean (0 invalid directives).")
    return total_anomalies

def main():
    parser = argparse.ArgumentParser(description="Headless Unsloth Trainer for 'batchcode' (Qwen2.5-Coder-1.5B)")
    parser.add_argument("--dataset", type=str, default="datacs_programming_dataset.jsonl", help="Path to .jsonl dataset")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Base HF model")
    parser.add_argument("--output-name", type=str, default="batchcode", help="Export model directory and Ollama name")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Context sequence length")
    parser.add_argument("--max-steps", type=int, default=60, help="Training steps (60 steps = ~3 epochs of 50 samples)")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--quant-method", type=str, default="q4_k_m", help="GGUF quantization method (e.g. q4_k_m, q8_0)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    args = parser.parse_args()

    # Step 0: Resolve dataset path
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        # Fallback alternatives
        alt_paths = [
            os.path.join("data", "cs_programming_dataset.jsonl"),
            os.path.join(os.path.dirname(__file__), "datacs_programming_dataset.jsonl"),
            os.path.join(os.path.dirname(__file__), "data", "cs_programming_dataset.jsonl")
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                dataset_path = alt
                break

    print(f"============================================================")
    print(f"  DistillLab -> Unsloth Headless Trainer: '{args.output_name}'")
    print(f"  Base Model:       {args.model_name}")
    print(f"  Dataset:          {dataset_path}")
    print(f"  Quantization:     4-bit QLoRA -> GGUF ({args.quant_method})")
    print(f"  Max Seq Length:   {args.max_seq_length}")
    print(f"============================================================")

    # Sanitize before training
    sanitize_dataset(dataset_path)

    # Import ML libraries after pre-flight checks
    try:
        import torch
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
    except ImportError as e:
        print("\n[ERROR] Unsloth / PyTorch dependencies missing.")
        print("Please install requirements using:")
        print("  pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
        print("  pip install --no-deps trl peft accelerate bitsandbytes\n")
        sys.exit(1)

    # 1. Load base model in 4-bit
    print(f"\n[1/5] Loading {args.model_name} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    # 2. Attach LoRA adapters to Attention + MLP layers
    print(f"[2/5] Attaching LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. Format dataset with ChatML template
    print(f"[3/5] Applying ChatML template to {dataset_path}...")
    def format_chatml(batch):
        texts = [tokenizer.apply_chat_template(convo, tokenize=False) for convo in batch["messages"]]
        return {"text": texts}

    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = raw_dataset.map(format_chatml, batched=True)
    print(f"Loaded {len(dataset)} samples for training.")

    # 4. Train with SFTTrainer
    steps = args.max_steps
    if steps <= 0 or (steps == 60 and len(dataset) > 50):
        # Auto-compute 3 epochs with effective batch size 8
        steps = max(60, int((len(dataset) * 3) / 8))

    print(f"[4/5] Starting headless SFTTrainer for {steps} steps (~3 epochs of {len(dataset)} samples)...")
    output_dir = f"./outputs_{args.output_name}"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=max(5, int(steps * 0.05)),
            max_steps=steps,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            output_dir=output_dir,
            seed=3407,
        ),
    )

    trainer_stats = trainer.train()
    print(f"Training completed successfully! Total runtime: {trainer_stats.metrics.get('train_runtime', 0):.2f}s")

    # 5. Export directly to Quantized GGUF for Ollama
    export_dir = args.output_name
    print(f"\n[5/5] Exporting Quantized GGUF ({args.quant_method}) to ./{export_dir}...")
    model.save_pretrained_gguf(export_dir, tokenizer, quantization_method=args.quant_method)
    print(f"GGUF quantization complete in ./{export_dir}/")

    # 6. Generate Ollama Modelfile
    modelfile_path = os.path.join(export_dir, "Modelfile")
    gguf_name = f"unsloth.{args.quant_method.upper()}.gguf"
    
    modelfile_content = f'''FROM ./{gguf_name}

TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1
'''

    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    print(f"Generated Ollama Modelfile: {modelfile_path}")

    # Instructions to load into Ollama
    print("\n" + "="*60)
    print(f"  SUCCESS! '{args.output_name}' is ready.")
    print("  To load and run in Ollama:")
    print(f"    cd {export_dir}")
    print(f"    ollama create {args.output_name} -f Modelfile")
    print(f"    ollama run {args.output_name}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
