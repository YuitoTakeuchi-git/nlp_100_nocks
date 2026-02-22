import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_multi_turn_chat():
    # Chat Templateに対応したGPT型モデル（TinyLlama）を使用
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # Define the conversation history
    # 94番の応答内容（Dessert）を履歴に含める
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What do you call a sweet eaten after dinner?"},
        {"role": "assistant", "content": "It is called a dessert."},
        {"role": "user", "content": "Please give me the plural form of the word with its spelling in reverse order."},
    ]

    # Apply the chat template to the entire history
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_file_path = os.path.join(out_dir, "095.txt")

    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write("--- Multi-turn Prompt with Chat Template ---\n")
        f.write(f"{prompt}\n")
        f.write("\n" + "="*50 + "\n\n")

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        # Decode only the newly generated part
        generated_ids = output_ids[0][len(input_ids[0]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        f.write("--- Model Response ---\n")
        f.write(f"{response.strip()}\n")

    print(f"Success: Multi-turn results saved to '{out_file_path}'")

if __name__ == "__main__":
    generate_multi_turn_chat()