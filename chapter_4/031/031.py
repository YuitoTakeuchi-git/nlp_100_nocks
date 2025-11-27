import spacy
import csv

# モデルの読み込み
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

# ファイルに書き込み
output_file = 'verbs_list.csv'

with open(output_file, 'w', newline='', encoding='utf-8') as f: # newline=''がないと、謎の空行が入る可能性有
    writer = csv.writer(f)

    writer.writerow(['元の単語', '基本形'])

    for token in doc:
        if token.pos_ == 'VERB': # pos = part of speach (品詞)
            writer.writerow([token.text, token.lemma_]) # lemma = 見出し語 辞書に載っている形

print(f'保存完了： {output_file}')    

