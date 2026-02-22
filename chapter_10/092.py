import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def calculate_sequence_probabilities():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_text = "The movie was full of"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    max_new_tokens = 5
    current_input_ids = input_ids

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_file_path = os.path.join(out_dir, "092.txt")

    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write(f"Initial Prompt: {input_text}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Generated Token':<20} | {'Probability':<12}\n")
        f.write("-" * 50 + "\n")

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(current_input_ids)
                
                next_token_logits = outputs.logits[0, -1, :]
                
                probs = F.softmax(next_token_logits, dim=-1)
                
                next_token_id = torch.argmax(probs).item()
                next_token_prob = probs[next_token_id].item()
                
                next_token_text = tokenizer.decode([next_token_id])
                
                f.write(f"{repr(next_token_text):<20} | {next_token_prob:.4f}\n")
                
                next_token_tensor = torch.tensor([[next_token_id]])
                current_input_ids = torch.cat([current_input_ids, next_token_tensor], dim=-1)

        full_sentence = tokenizer.decode(current_input_ids[0], skip_special_tokens=True)
        f.write("-" * 50 + "\n")
        f.write(f"Final Sentence: {full_sentence}\n")

    print(f"Success: Results saved to '{out_file_path}'")

if __name__ == "__main__":
    calculate_sequence_probabilities()