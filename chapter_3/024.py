import re
import pandas as pd

basepath = '/home/takeuchi/workspace/nlp_100_nocks/chapter_3'

df = pd.read_json('{}/jawiki-country.json'.format(basepath), lines=True)

uk_text = df.query('title=="イギリス"')["text"].values[0]
for file in re.findall(r"\[\[(ファイル|File):([^]|]+?)(\|.*?)+\]\]", uk_text):
    print(file[1])

