import spacy

nlp = spacy.load('ja_ginza')

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

doc = nlp(text)

output_file = 'dependency_list.txt'

with open(output_file, "w", encoding='utf-8') as f:
    f.write(f"係り元(自分)\t係り先(親)\n")
    f.write("-" * 30 + "\n")

    for token in doc:
        line = f"{token.text}\t{token.head.text}\n"
        f.write(line)

print(f"保存完了： {output_file}")