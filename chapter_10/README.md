# chapter_10

この章では、LLM（主に GPT-2 / TinyLlama）を使って、
**推論の基礎** から **生成制御**、**評価**、**軽量ファインチューニング（LoRA / DPO）** までを段階的に扱います。

## 概要

- 前半（090〜095）: 次トークン予測、デコーディング手法比較、確率・パープレキシティ、チャットテンプレート
- 後半（096〜099）: SST-2 感情分類でゼロショット評価、分類ヘッド学習、LoRA学習、DPO学習
- 結果出力は主に `out/`、学習済みモデルは `models/` に保存

## 各課題

### 090: 次トークン予測
- プロンプト: `The movie was full of`
- トークン化結果を確認
- 次トークン候補 Top 10 と確率を出力
- 出力: `out/090.txt`

### 091: 生成設定の比較
- Greedy Search
- Beam Search (`num_beams=5`)
- Temperature sampling (`temp=2.0`, `temp=0.5`)
- Top-p sampling (`p=0.9`)
- 出力: `out/091.txt`

### 092: 逐次生成と確率
- 1トークンずつ5回生成し、各ステップの選択確率を記録
- 最終文を復元して表示
- 出力: `out/092.txt`

### 093: パープレキシティ計測
- 複数文に対して loss と perplexity を比較
- 文法誤りを含む文との相対比較を実施
- 出力: `out/093.txt`

### 094: チャットテンプレート（単一ターン）
- TinyLlama の chat template を適用
- 整形済みプロンプトと応答を保存
- 出力: `out/094.txt`

### 095: チャットテンプレート（複数ターン）
- 会話履歴を含む multi-turn プロンプトで応答生成
- 出力: `out/095.txt`

### 096: ゼロショット感情分類（SST-2）
- 指示プロンプトで `positive` / `negative` を直接生成
- 100件で簡易評価
- 出力: `out/096.txt`

### 097: 埋め込み＋線形分類ヘッド
- ベースLLMを固定し、最終隠れ状態から2値分類
- 学習: train 1000件、評価: validation 100件
- 出力: `out/097.txt`

### 098: LoRAによるファインチューニング
- TinyLlama に LoRA（`q_proj`, `v_proj`）を適用して学習
- 学習済み重み保存: `models/098_sentiment_lora`
- 評価結果: `out/098.txt`

### 099: DPOによる選好最適化
- SST-2 から chosen / rejected ペアを作成して DPO 学習
- 学習済み重み保存: `models/099_sentiment_dpo`
- 評価結果: `out/099.txt`

## 実行メモ

- 各スクリプトは単体実行可能（例: `python 090.py`）
- 依存関係は `pyproject.toml` を参照
- GPU利用を前提とした設定（`torch.float16`, `device_map="auto"`）を含むため、環境によっては調整が必要
