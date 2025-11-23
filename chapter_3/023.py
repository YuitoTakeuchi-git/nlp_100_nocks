import json
import pandas as pd
import re

basepath = '/home/takeuchi/workspace/nlp_100_nocks/chapter_3'

df = pd.read_json('{}/jawiki-country.json'.format(basepath), lines=True)

uk_text = df.query('title=="イギリス"')["text"].values[0]
for section in re.findall(r"(=+)([^=]+)\1\n", uk_text): #タプル形式でマッチしたものが入っていく。('==', '歴史')
    print(f"{section[1].strip()}\t{len(section[0]) - 1}") #レベルとは、=の数
