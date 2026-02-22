import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_texts_with_configs():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_text = "The movie was full of"
    input_ids = tokenizer.encode(input_text, return_tensors="pt") # tensor型に変換

    configs = [
        ("Greedy Search", {"do_sample": False}),
        ("Beam Search (num_beams=5)", {"do_sample": False, "num_beams": 5}),
        ("High Temperature (temp=2.0)", {"do_sample": True, "temperature": 2.0}),
        ("Low Temperature (temp=0.5)", {"do_sample": True, "temperature": 0.5}),
        ("Top-p Sampling (p=0.9)", {"do_sample": True, "top_p": 0.9})
    ]

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_file_path = os.path.join(out_dir, "091.txt")

    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {input_text}\n\n")

        for name, params in configs:
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=20,
                **params
            )

            generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            f.write(f"--- {name} ---\n")
            f.write(f"{generated_text}\n\n")

    print(f"Success: Analysis results saved to '{out_file_path}'")


if __name__ == "__main__":
    generate_texts_with_configs()