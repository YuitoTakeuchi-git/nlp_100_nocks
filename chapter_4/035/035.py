import spacy
from spacy import displacy
from pathlib import Path

nlp = spacy.load('ja_ginza')

text = 'メロスは激怒した。'
doc = nlp(text)

options = {
    "compact": True, # コンパクトに表示
    "bg": "#ffffff", # 背景を白に
    "color": "black", #文字色を黒に
    "font": "Source Han Sans" #日本語フォント指定
}

svg = displacy.render(doc, style="dep", jupyter=False, options=options)

output_path = Path("melos_tree.svg")
with open(output_path, "w", encoding='utf-8') as f:
    f.write(svg)

print(f"保存完了： {output_path.absolute()}")