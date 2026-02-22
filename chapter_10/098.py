import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

def run_98_full():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 1. モデルの読み込みとLoRAの設定
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    lora_config = LoraConfig(
        r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 2. データセットの準備
    dataset = load_dataset("glue", "sst2")
    
    def preprocess(examples):
        # 学習用の形式：入力と正解をセットにする
        prompts = [f"<|user|>\nAnalyze: {t}</s>\n<|assistant|>\n{'positive' if l==1 else 'negative'}</s>" 
                   for t, l in zip(examples["sentence"], examples["label"])]
        return tokenizer(prompts, truncation=True, max_length=128, padding="max_length")

    train_data = dataset["train"].select(range(1000)).map(preprocess, batched=True)

    # 3. 学習の実行
    args = TrainingArguments(
        output_dir="./res_98", per_device_train_batch_size=4, num_train_epochs=1, 
        learning_rate=2e-4, fp16=True, save_strategy="no", report_to="none"
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_data, 
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    print("Starting Fine-tuning (98)...")
    trainer.train()

    # 4. モデルの保存（重みファイル）
    os.makedirs("models", exist_ok=True)
    model.save_pretrained("./models/098_sentiment_lora")

    # 5. 推論結果をテキストファイルに保存
    os.makedirs("out", exist_ok=True)
    out_file_path = "out/098.txt"
    
    model.eval()
    correct = 0
    total = 0
    eval_results = []
    
    val_data = dataset["validation"].select(range(100))
    print("Evaluating and saving to 098.txt...")

    for item in tqdm(val_data):
        text = item["sentence"]
        true_label = item["label"]
        
        # 推論用のプロンプト（アシスタントの回答直前まで）
        input_text = f"<|user|>\nAnalyze: {text}</s>\n<|assistant|>\n"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2, do_sample=False)
            # 生成された部分だけをデコード
            pred_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()

        pred_label = 1 if "positive" in pred_text else 0
        if pred_label == true_label:
            correct += 1
        total += 1
        
        eval_results.append(f"Text: {text}\nTrue: {true_label}, Pred: {pred_label} ({pred_text})\n\n")

    accuracy = correct / total
    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write(f"--- 98. Fine-tuning (LoRA) Sentiment Analysis ---\n")
        f.write(f"Accuracy: {accuracy:.4f} ({correct}/{total})\n")
        f.write("-" * 50 + "\n\n")
        f.writelines(eval_results)

    print(f"Success: Results saved to '{out_file_path}'")

if __name__ == "__main__":
    run_98_full()