import spacy
import csv

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

output_file = 'melos_predicates.csv'

with open(output_file, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    writer.writerow(['主語', '述語'])

    for token in doc:
        if token.text == 'メロス' and token.dep_ == "nsubj": #nsubj = 名詞の述語、token.dep_ =.単語の役割
            subject = token.text
            predicate = token.head.text # token.head = 述語

            writer.writerow([subject, predicate])

print(f"記述完了：{output_file}")