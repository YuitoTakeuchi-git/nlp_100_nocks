import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_chat_response():
    # Chat Templateに対応したGPT型モデルを指定（ここでは軽量なTinyLlamaを使用）
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # Define the conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What do you call a sweet eaten after dinner?"},
    ]

    # Apply the chat template
    # tokenize=False to see the raw text prompt, add_generation_prompt=True to append the assistant header
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # ここではまず、変換された「生のテキストプロンプト」を確認するために、数値化せず文字列のまま取得している。
    # add_generation_prompt=True: プロンプトの最後に「AIの回答がここから始まる」ことを示す特殊トークン（ヘッダー）を強制的に付与する。

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_file_path = os.path.join(out_dir, "094.txt")

    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write("--- Prepared Prompt with Chat Template ---\n")
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

        # Decode only the generated part
        generated_ids = output_ids[0][len(input_ids[0]):]
        # output_ids[0][len(input_ids[0]):]: 出力されたID列から、入力プロンプト部分を差し引き、新しく生成された回答部分のみを抽出する。
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # skip_special_tokens=True は、トークナイザーがテキストを復元（デコード）する際に、モデルの制御用に使われる「特殊なトークン」を無視して、人間が読むための地文だけを表示させるための設定である。

        f.write("--- Model Response ---\n")
        f.write(f"{response.strip()}\n")

    print(f"Success: Prompt and response saved to '{out_file_path}'")

if __name__ == "__main__":
    generate_chat_response()