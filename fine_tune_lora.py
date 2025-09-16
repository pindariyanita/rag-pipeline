# fine_tune_lora.py
import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

MODEL_NAME = "tiiuae/falcon-7b-instruct"  # pick smaller if needed
TOKENIZER_NAME = MODEL_NAME
OUTPUT_DIR = "lora_finetuned"

def preprocess_fn(examples):
    # expects dataset of {"instruction": "", "input":"", "output":""}
    texts = []
    for ins, inp, out in zip(examples["instruction"], examples.get("input", [""]*len(examples["instruction"])), examples["output"]):
        prompt = f"### Instruction:\n{ins}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        texts.append(prompt)
    return {"text": texts}

def main():
    # load your dataset - replace path/dataset name
    ds = load_dataset("json", data_files={"train":"train.jsonl","validation":"valid.jsonl"})
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto", trust_remote_code=True)

    # Tokenize
    def tok(ex):
        return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=1024)
    ds = ds.map(preprocess_fn, batched=True)
    ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=1024), batched=True)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        data_collator=data_collator
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
