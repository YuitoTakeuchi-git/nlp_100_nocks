import re
import pandas as pd

basepath = "/home/takeuchi/workspace/nlp_100_nocks/chapter_3"

def remove_mk(v):
    r1 = re.compile("'+")
    r2 = re.compile("\[\[(.+\||)(.+?)\]\]")
    r3 = re.compile("\{\{(.+\||)(.+?)\}\}")
    r4 = re.compile("<\s*?/*?\s*?br\s*?/*?\s*>")
    v = r1.sub("", v)
    v = r2.sub(r"\2", v)
    v = r3.sub(r"\2", v)
    v = r4.sub("", v)
    return v


df = pd.read_json('{}/jawiki-country.json'.format(basepath), lines=True)
uk_text = df.query('title=="イギリス"')["text"].values[0]
uk_texts = uk_text.split('\n')

pattern = re.compile("\|(.+?)\s=\s*(.+)")

ans = {}

for line in uk_texts:
    r = re.search(pattern, line)
    if r:
        ans[r[1]] = r[2]

ans = {k: remove_mk(v) for k, v in ans.items()}
print(ans)