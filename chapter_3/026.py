import re
import pandas as pd

def remove_stress(dc):
    r = re.compile("'+")
    return {k: r.sub("", v) for k, v in dc.items()}

basepath = '/home/takeuchi/workspace/nlp_100_nocks/chapter_3'
df = pd.read_json('{}/jawiki-country.json'.format(basepath), lines=True)
uk_text = df.query('title=="イギリス"')["text"].values[0]
uk_texts = uk_text.split('\n')

ans = {}

pattern = re.compile("\|(.+?)\s=\s*(.+)")

for line in uk_texts:
    r = re.search(pattern, line)
    if r:
        ans[r[1]] = r[2]

print(remove_stress(ans))