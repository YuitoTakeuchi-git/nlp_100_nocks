import json
import pandas as pd

basepath = '/home/takeuchi/workspace/nlp_100_nocks/chapter_3/'

df = pd.read_json('{}/jawiki-country.json'.format(basepath), lines=True)
uk_text = df.query('title=="イギリス"')["text"].values[0] #[0]を使って、配列ではなくただの文字列に変換している。
uk_text = uk_text.split('\n')
ans = list(filter(lambda x: "[Category:" in x, uk_text))


print(ans)


    

