# パープレキシティは「次に続く単語の選択肢が平均して何個に絞られているか」を示す指標

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def calculate_perplexity():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    sentences = [
        "The movie was full of surprises",
        "The movies were full of surprises",
        "The movie were full of surprises", # Grammatical error (Singular + Were)
        "The movies was full of surprises"  # Grammatical error (Plural + Was)
    ]

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_file_path = os.path.join(out_dir, "093.txt")

    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write("--- Perplexity Measurement ---\n\n")

        for sentence in sentences:
            input_ids = tokenizer.encode(sentence, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                # GPT-2 in Hugging Face calculates Cross Entropy Loss automatically if 'labels' are provided.
                loss = outputs.loss
                # Perplexity is the exponential of the average loss.
                perplexity = torch.exp(loss).item()

            f.write(f"Sentence: {sentence}\n")
            f.write(f"Loss: {loss.item():.4f}\n")
            f.write(f"Perplexity: {perplexity:.4f}\n")
            f.write("-" * 30 + "\n")

    print(f"Success: Analysis results saved to '{out_file_path}'")

if __name__ == "__main__":
    calculate_perplexity()