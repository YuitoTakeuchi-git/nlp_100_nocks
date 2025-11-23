import json
import pandas as pd

basepath = '/home/takeuchi/workspace/nlp_100_nocks/chapter_3'

df = pd.read_json('{}/jawiki-country.json'.format(basepath), lines=True)
uk_text = df.query('title=="イギリス"')["text"].values[0]
uk_text = uk_text.split('\n') #改行で区切る配列にする。
ans = list(filter(lambda x: "[Category:" in x, uk_text))
ans = [a.replace("[[Category:", "").replace("|*", "").replace(']]', "") for a in ans]
print(ans)