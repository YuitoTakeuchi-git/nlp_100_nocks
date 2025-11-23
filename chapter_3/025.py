import re
import pandas as pd

basepath = '/home/takeuchi/workspace/nlp_100_nocks/chapter_3'

df = pd.read_json('{}/jawiki-country.json'.format(basepath), lines=True)
uk_text = df.query('title=="イギリス"')["text"].values[0]
uk_text = uk_text.split('\n')

pattern = re.compile("\|(.+?)\s=\s*(.+)") # ?があると、非貪欲マッチになる。これがないと、=まで飲み込んでしまう。

ans = {}

for line in uk_text:
    r = re.search(pattern, line)
    if r:
        ans[r[1]] = r[2] #ans["略称"] = "イギリス"
print(ans)