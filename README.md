# Poet

机器学习概论大作业。项目分两套数据和两套逻辑：

- 诗：唐诗、宋诗。
- 词：宋词，有词牌、词风、小令/中调/长调。

分类、聚类、生成都要先选 `poem` 还是 `ci`。不能把诗和词混在一起跑。

## 0. 现在的标签约定

`dataset/llm_labels` 是唯一真实标签目录。这个目录里的标签必须是大语言模型逐首阅读后填写的结果。

如果没有 `dataset/llm_labels/train_labels.jsonl` 和 `dataset/llm_labels/test_labels.jsonl`，`classifier.py` 和 `cluster.py` 会直接停止，不会偷偷使用规则标签。

## 1. 环境

```bash
cd C:\Users\liuqi\Desktop\机器学习概论\Poet
pip install requests beautifulsoup4 opencc-python-reimplemented pypinyin scikit-learn scipy numpy kmedoids gensim streamlit
```

## 2. 数据

第一次爬数据：

```bash
python crawler.py
```

少量测试：

```bash
python crawler.py --max_file 2
```

输出：

```text
dataset/poems/train.jsonl
dataset/poems/vaid.jsonl
dataset/poems/test.jsonl
dataset/rhyme/cipai_len.json
dataset/tone/char_tone.json
```

## 3. LLM 真实标签

现在真实标签用 LiteLLM API 逐首标。key 不写在代码里，先在命令行里设环境变量：

```bash
set LITELLM_KEY=你的key
```

PowerShell 用：

```bash
$env:LITELLM_KEY="你的key"
```

从头重标一份诗的 train：

```bash
python llm_label_api.py --split train --kind poem --fresh --limit 0 --model gemini-3-flash-preview
```

从头重标一份词的 train：

```bash
python llm_label_api.py --split train --kind ci --fresh --limit 0 --model gemini-3-flash-preview
```

`--fresh` 会删掉当前 split 里同一种类的旧标签，只保留另一种。比如 `--kind poem --fresh` 只删诗标签，不删词标签。`--limit 0` 表示一直跑到这个 split 结束。

从头重标全部诗和词：

```bash
python llm_label_api.py --split all --kind all --fresh --limit 0 --model gemini-3-flash-preview
```

断点继续时不要加 `--fresh`：

```bash
python llm_label_api.py --split train --kind poem --limit 0
```

少量检查：

```bash
python llm_label_api.py --split test --kind ci --limit 3
```

旧的 `llm_label.py` 只用来生成待标注文件和检查数量：

```bash
python llm_label.py --make --split train --kind poem --start 0 --limit 50
python llm_label.py --make --split train --kind ci --start 0 --limit 50
```

输出在：

```text
dataset/llm_todo/poem_train_0.jsonl
dataset/llm_todo/ci_train_0.jsonl
```

这里面的 `theme/season/festival/emotion/style/ci_style/form` 需要由 LLM 逐首阅读后填写。每一条都要看原文，能归入类别就归入类别，只有真的落不到任何类别时才写 `N/A`。填完后的标签放到：

```text
dataset/llm_labels/train_labels.jsonl
dataset/llm_labels/vaid_labels.jsonl
dataset/llm_labels/test_labels.jsonl
```

检查标签数量：

```bash
python llm_label.py --check --kind poem
python llm_label.py --check --kind ci
```

`label.py` 只保存分类标准、诗体判断、韵脚和平仄辅助函数，不负责自动标注。

## 4. 分类命令行

诗分类实验：

```bash
python classifier.py --kind poem --standard all --model all --train_limit 3000 --test_limit 1000 --save
```

词分类实验：

```bash
python classifier.py --kind ci --standard all --model all --train_limit 3000 --test_limit 1000 --save
```

诗和词一起跑：

```bash
python classifier.py --kind all --standard all --model all --train_limit 3000 --test_limit 1000 --save
```

只跑一个标准、一个模型：

```bash
python classifier.py --kind poem --standard theme --model SVM
python classifier.py --kind ci --standard ci_style --model Tree
```

命令行输入一首诗并分类：

```bash
python classifier.py --predict --kind poem --model best --standard all --text "君过秋浦正逢秋，亦到枞阳皖水头。九派先将明月去，三峰少为白云留。"
```

命令行输入一首词并分类：

```bash
python classifier.py --predict --kind ci --cipai 浣溪沙 --model best --standard all --text "一曲新词酒一杯，去年天气旧亭台。夕阳西下几时回。"
```

`--model best` 会按当前设置选默认最好模型。也可以换成：

```text
KNN, WKNN, Bayes, SVM, Tree,
Bagging_KNN, Bagging_WKNN, Bagging_Bayes, Bagging_SVM, Bagging_Tree,
AdaBoost_Bayes, AdaBoost_SVM, AdaBoost_Tree
```

分类实验表写到：

```text
dataset/results/classifier_report.md
```

分类输出字段：

- 诗：`theme/season/festival/emotion/style/form`
- 词：`theme/season/festival/emotion/style/ci_style/form`

## 5. 聚类命令行

诗聚类：

```bash
python cluster.py --kind poem --standard all --method all --limit 300
```

词聚类：

```bash
python cluster.py --kind ci --standard all --method all --limit 300
```

只跑一个聚类算法：

```bash
python cluster.py --kind poem --standard theme --method kmeans --limit 300
python cluster.py --kind ci --standard ci_style --method pam --limit 300
```

可选方法：

```text
kmeans, kmedoids, pam, agglomerative, bisect
```

结果写到：

```text
dataset/results/clusters/poem_*.json
dataset/results/clusters/ci_*.json
dataset/results/clusters/summary_poem.md
dataset/results/clusters/summary_ci.md
```

聚类只用来验证算法能不能逼近 LLM 分类结果。每个簇会按簇内最多的 LLM 标签贴一个名字，再计算 `acc/f1/silhouette`。

## 6. 生成命令行

生成诗：

```bash
python generator.py --kind poem --form 七言绝句 --theme 山水田园 --emotion 思乡怀人 --rhyme an
```

生成词：

```bash
python generator.py --kind ci --cipai 浣溪沙 --ci_style 婉约清丽 --emotion 离愁别绪 --word 月,柳
```

生成器按小句生成，不把整首诗当成一个长字符串。

## 7. Word2Vec

训练：

```bash
python word2vec.py --limit 120000 --size 50 --epochs 5
```

然后用 embedding 重跑分类/聚类：

```bash
python classifier.py --kind poem --standard all --model all --use_w2v
python cluster.py --kind ci --standard all --method all --use_w2v
```

## 8. 全量重跑

拿到全部 LLM 标签后，用这个命令重新跑分类和聚类：

```bash
python run_all.py --cluster_limit 1200
```

它会先检查 `train/vaid/test` 的诗和词标签数量。没齐就只打印缺多少，不会跑。

分类会跑普通特征和 word2vec 特征两套：

```text
dataset/results/classifier_report.md
dataset/results/classifier_report_w2v.md
```

聚类也会跑诗、词、普通特征、word2vec 特征。`cluster_limit` 是聚类采样上限，因为 PAM、K-medoids、凝聚聚类不能直接吃几十万首。

## 9. App

启动：

```bash
streamlit run app.py
```

页面：

- 分类：先选诗/词，再填文本。词要填词牌。默认 `best` 分类器，也可以手动换模型和分类标准。
- 诗词大全：先选诗/词，再按分类标准和标签筛选作品。
- 生成：先选诗/词。诗选诗体；词填词牌；可填题材、情感、风格、韵脚、关键词。
- 报告：看分类实验报告和诗/词各自的聚类表。

如果没有 `dataset/llm_labels`，分类页和诗词大全页会提示缺少 LLM 标签。

## 10. 文件说明

- `crawler.py`：爬数据，切 train/vaid/test。
- `llm_label.py`：生成待 LLM 标注文件，检查 LLM 标签是否齐全。
- `llm_label_api.py`：调用 LiteLLM API 逐首生成 true label。
- `label.py`：分类标准和格式辅助函数，不自动生成 true label。
- `config.py`：诗/词分类标准、特征、路径。
- `classifier.py`：按诗/词分别训练、测试、预测。
- `cluster.py`：按诗/词分别聚类并和 LLM 标签比较。
- `generator.py`：按诗/词分别生成。
- `word2vec.py`：训练字级 embedding。
- `run_all.py`：标签齐全后重跑分类和聚类。
- `app.py`：前端。
- `report.md`：本地报告，不进 git。
