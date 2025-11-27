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

output_file = 'verbs_list.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    for token in doc:
        if token.pos_ == 'VERB':
            writer.writerow([token.text])

    print(f'アウトプット完了：{output_file}')
