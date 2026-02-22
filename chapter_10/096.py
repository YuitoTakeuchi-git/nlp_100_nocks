import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

def evaluate_sentiment_analysis():
    # Load the SST-2 development dataset
    dataset = load_dataset("glue", "sst2", split="validation")
    
    # Use a chat-tuned or instruction-tuned model for zero-shot performance
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    correct = 0
    total = 0
    
    # Process only a subset for quick evaluation (e.g., 100 samples)
    eval_limit = 100
    results = []

    for i in tqdm(range(eval_limit)):
        item = dataset[i]
        text = item["sentence"]
        label = item["label"] # 1: positive, 0: negative

        # Construct the prompt with instructions
        messages = [
            {"role": "system", "content": "Analyze the sentiment of the text. Respond with only one word: 'positive' or 'negative'."},
            {"role": "user", "content": f"Text: {text}\nSentiment:"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=2, do_sample=False)
            generated_text = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True).lower()
            # .lower()で小文字に変換。

        # Simple string matching for evaluation
        predicted_label = 1 if "positive" in generated_text else 0
        if predicted_label == label:
            correct += 1
        total += 1
        
        results.append(f"Text: {text}\nTrue: {label}, Pred: {predicted_label} ({generated_text.strip()})\n")

    accuracy = correct / total

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_file_path = os.path.join(out_dir, "096.txt")

    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})\n\n")
        f.writelines(results)

    print(f"Success: Accuracy is {accuracy:.4f}. Results saved to '{out_file_path}'")

if __name__ == "__main__":
    evaluate_sentiment_analysis()