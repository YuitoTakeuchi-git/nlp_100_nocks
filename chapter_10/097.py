import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# モデル定義は前回と同様
class SentimentClassifier(nn.Module):
    def __init__(self, base_model):
        super(SentimentClassifier, self).__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
            last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        
        logits = self.fc(last_hidden_state.to(self.fc.weight.device).to(self.fc.weight.dtype))
        return logits

def train_and_save_results():
    # 準備（モデル・トークナイザー）
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    
    classifier = SentimentClassifier(base_model)
    output_device = base_model.lm_head.weight.device
    classifier.fc.to(output_device).to(torch.float16)

    # データセット（SST-2）
    dataset = load_dataset("glue", "sst2")
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=64)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    train_loader = DataLoader(tokenized_dataset["train"].select(range(1000)), batch_size=16, shuffle=True)
    val_loader = DataLoader(tokenized_dataset["validation"].select(range(100)), batch_size=1) # 評価用100件

    optimizer = optim.Adam(classifier.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 学習
    classifier.train()
    for epoch in range(1):
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(base_model.device)
            attention_mask = batch["attention_mask"].to(base_model.device)
            labels = batch["label"].to(classifier.fc.weight.device)
            outputs = classifier(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # --- 評価とファイル保存 ---
    classifier.eval()
    correct = 0
    total = 0
    eval_results = []

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_file_path = os.path.join(out_dir, "097.txt")

    print("Evaluating and saving results...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            input_ids = batch["input_ids"].to(base_model.device)
            attention_mask = batch["attention_mask"].to(base_model.device)
            labels = batch["label"].to(classifier.fc.weight.device)

            logits = classifier(input_ids, attention_mask)
            prediction = torch.argmax(logits, dim=-1)
            
            # 元のテキストを復元
            text = dataset["validation"][i]["sentence"]
            true_label = labels.item()
            pred_label = prediction.item()

            if true_label == pred_label:
                correct += 1
            total += 1

            eval_results.append(f"Text: {text}\nTrue: {true_label}, Pred: {pred_label}\n\n")

    accuracy = correct / total
    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write(f"--- Embedding-based Sentiment Analysis ---\n")
        f.write(f"Accuracy: {accuracy:.4f} ({correct}/{total})\n")
        f.write("-" * 50 + "\n\n")
        f.writelines(eval_results)

    print(f"Success: Results saved to '{out_file_path}'")

if __name__ == "__main__":
    train_and_save_results()