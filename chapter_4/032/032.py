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

noun_phrases = []

for i in range(1, len(doc) - 1):
    token = doc[i]
    prev_token = doc[i - 1]
    next_token = doc[i + 1]

    if token.text == 'の':
        target_pos = ['NOUN', 'PROPN', 'PRON']

        if prev_token.pos_ in target_pos and next_token.pos_ in target_pos:
            phrase = f"{prev_token.text}の{next_token.text}"
            noun_phrases.append(phrase)

print("抽出された名詞句：", noun_phrases)

output_file = 'noun_phrases.csv'
with open(output_file, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['名詞句'])

    for phrase in noun_phrases:
        writer.writerow([phrase])

print(f'{output_file}に保存しました。')

