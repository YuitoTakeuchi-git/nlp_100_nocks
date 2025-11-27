import gzip
import json
import re
import spacy
import csv
from collections import Counter

nlp = spacy.load('ja_ginza', disable=['ner', 'parser']) #今回は固有表現抽出や係り受け解析は不要のため

def remove_markup(text):
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]+\|)?([^|\]]+)\]\]", r"\1", text) #?:で、グループとしてカウントする必要がないという意味。[]の中で、特殊クラスはその効果を失う。
    text = re.sub(r"\[http.+?\]", "", text)
    text = re.sub(r"\{\{.+?\}\}", "", text)
    text = re.sub(r"={2,}", "", text)
    text = re.sub(r"<.+?>", "", text)
    return text

def read_wiki_corpus(filename):
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                raw_text = data.get('text', '')
                clean_text = remove_markup(raw_text)

                for sentence in clean_text.split('\n'):
                    if not sentence.strip():
                        continue

                    yield sentence # yieldは、出力をリストにするのではなく、一つ処理できたら渡すという処理を逐次的に行う。

            except json.JSONDecodeError:
                continue

input_file = '../jawiki-country.json.gz'
output_file = 'word_freq_top20.csv'
word_freq = Counter() # データを渡すだけで、どれが何個あるかを辞書形式で管理してくれる。

for doc in nlp.pipe(read_wiki_corpus(input_file), batch_size=50):
    tokens = [
        token.lemma_ for token in doc
        if token.pos_ not in ['PUNCT', 'SYM', 'SPACE']
    ]
    word_freq.update(tokens)

with open(output_file, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['順位', '単語', '出現回数'])
    for rank, (word, count) in enumerate(word_freq.most_common(20), 1): #1番目から始めるという意味
        print(f"{rank:<5} | {word:<15} | {count}")
        writer.writerow([rank, word, count])

print("完了")

#yieldの方がメモリ効率が良い