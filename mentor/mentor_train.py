"""QLoRA fine-tune of Qwen2.5-7B-Instruct on the punchy music-mentor voice.

Trains on assistant tokens only (completion_only), 4-bit NF4 base + LoRA adapter.
"""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from paths import BASE_DIR as MODEL_DIR, LORA_DIR as OUT_DIR, TRAIN_JSONL, VAL_JSONL

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="eager",
    )
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )

    train_ds = load_dataset("json", data_files=TRAIN_JSONL, split="train")
    val_ds   = load_dataset("json", data_files=VAL_JSONL,   split="train")

    args = SFTConfig(
        output_dir=OUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,      # eff. batch 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        max_length=1024,
        packing=False,
        assistant_only_loss=True,           # loss on assistant turns only
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    trainer = SFTTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok, peft_config=peft_cfg,
    )
    trainer.train()

    # save adapter + tokenizer
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    # dump loss history
    hist = trainer.state.log_history
    with open(os.path.join(OUT_DIR, "log_history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"PEAK_TRAIN_VRAM_GB={peak:.2f}")
    print("DONE")

if __name__ == "__main__":
    main()
