import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig
from tqdm import tqdm

def run_99_full():
    # 1. 環境準備
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # モデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    
    # 2. LoRAの設定
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        task_type="CAUSAL_LM"
    )

    # 3. データセットの準備
    dataset = load_dataset("glue", "sst2")
    
    def create_dpo_dataset(examples):
        # 学習時のプロンプトも推論時に合わせる
        prompts = [f"<|user|>\nAnalyze the sentiment: {t}</s>\n<|assistant|>\nThe sentiment is " for t in examples["sentence"]]
        chosen = ["positive</s>" if l == 1 else "negative</s>" for l in examples["label"]]
        rejected = ["negative</s>" if l == 1 else "positive</s>" for l in examples["label"]]
        return {"prompt": prompts, "chosen": chosen, "rejected": rejected}

    # 学習データ（1000件に増強して安定化を図る）
    train_dpo_data = dataset["train"].select(range(1000)).map(create_dpo_dataset, batched=True)

    # 4. DPO学習の設定
    args = DPOConfig(
        output_dir="./res_99", 
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        num_train_epochs=1, 
        learning_rate=1e-4, # 小さなモデル向けに少し高めに設定
        beta=0.1, 
        fp16=True, 
        save_strategy="no", 
        report_to="none"
    )
    
    trainer = DPOTrainer(
        model=model, 
        args=args, 
        train_dataset=train_dpo_data, 
        processing_class=tokenizer,
        peft_config=lora_config
    )
    
    print("DPO学習を開始する...")
    trainer.train()

    # 5. 保存
    os.makedirs("models", exist_ok=True)
    trainer.save_model("./models/099_sentiment_dpo")

    # 6. 推論と評価（強制誘導プロンプト）
    os.makedirs("out", exist_ok=True)
    out_file_path = "out/099.txt"
    
    eval_model = trainer.model
    eval_model.eval()
    
    correct = 0
    total = 0
    eval_results = []
    
    val_data = dataset["validation"].select(range(100))
    print("評価を実行中...")

    for item in tqdm(val_data):
        text = item["sentence"]
        true_label = item["label"]
        
        # 文末を "The sentiment is " で終わらせることで、次に来る単語を positive/negative に固定する
        prompt = f"<|user|>\nAnalyze the sentiment: {text}</s>\n<|assistant|>\nThe sentiment is "
        inputs = tokenizer(prompt, return_tensors="pt").to(eval_model.device)
        
        with torch.no_grad():
            outputs = eval_model.generate(
                **inputs, 
                max_new_tokens=1, # 1単語だけ生成
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            # 生成された単語を抽出
            pred_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()

        # 判定
        pred_label = 1 if "positive" in pred_text else 0
        if pred_label == true_label:
            correct += 1
        total += 1
        
        eval_results.append(f"文章: {text}\n正解: {true_label}, 予測: {pred_label} (出力: {pred_text})\n\n")

    accuracy = correct / total
    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write(f"--- 99. 選好最適化 (DPOプロンプト制御版) ---\n")
        f.write(f"正解率: {accuracy:.4f} ({correct}/{total})\n")
        f.write("-" * 50 + "\n\n")
        f.writelines(eval_results)

    print(f"完了：'{out_file_path}' に結果を保存した。正解率: {accuracy:.4f}")

if __name__ == "__main__":
    run_99_full()