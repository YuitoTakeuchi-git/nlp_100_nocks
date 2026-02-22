import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def predict_next_token_and_save():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_text = "The movie was full of"

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    out_file_path = os.path.join(out_dir, "090.txt")

    with open(out_file_path, "w", encoding="utf-8") as f:
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        f.write("Tokenization Check\n")
        for token in tokens:
            f.write(f"{token}\n")

        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[0, -1, :] # 3次元。[batch_size, sequence_length, vocav_size] 一つの文章だけなので、batch_sizeは1。

        probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(probabilities, 10)

        f.write("\n--- Top 10 Next Tokens ---\n")
        for i in range(10):
            token_id = top_k_indices[i].item()
            token_text = tokenizer.decode([token_id])
            prob = top_k_probs[i].item()
            f.write(f"Rank {i+1}: '{token_text}' (Probability: {prob:.4f})\n")

        print(f"Success: Results have been saved to '{out_file_path}'")
    
if __name__ == "__main__":
    predict_next_token_and_save()
