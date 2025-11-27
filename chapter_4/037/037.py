import json
import spacy
import csv
import re
import gzip
from collections import Counter

nlp = spacy.load('ja_ginza', disable=['ner', 'parser'])

def remove_markup(text):
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]+\|)?([^|\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[http.+?\]", "", text)
    text = re.sub(r"\{\{.+?\}\}", "", text)
    text = re.sub(r"={2,}", "", text)
    text = re.sub(r"<.+?>", "", text)
    return text

def read_wiki_corpus(filename):
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line) #入力は文字列。出力は辞書。
                raw_text = data.get('text', '')
                clean_text = remove_markup(raw_text)

                for sentence in clean_text.split('\n'):
                    if not sentence.strip():
                        continue
                    
                    yield sentence

            except json.JSONDecodeError:
                continue

input_file = '../jawiki-country.json.gz'
output_file = 'noun_freq_top20.csv'
word_freq = Counter()
stop_words = ['|', '}', '{', '%', 'file', 'File']

for doc in nlp.pipe(read_wiki_corpus(input_file), batch_size=50):
    tokens = [
        token.lemma_ for token in doc
        if token.pos_ in ['NOUN', 'PROPN'] and token.text not in stop_words
    ]
    word_freq.update(tokens)

with open(output_file, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['順位', '単語', '出現回数'])

    for rank, (word, count) in enumerate(word_freq.most_common(20), 1):
        writer.writerow([rank, word, count])

print("完了")