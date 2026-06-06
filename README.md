# Poet

机器学习概论大作业。数据用唐诗、宋诗、宋词两套任务面：诗和词都做分类、聚类和生成，宋词另外加词牌和词风。

代码尽量写得朴素一点：爬虫直接爬，分类器和聚类器直接调用 sklearn / kmedoids，特征也用人工关键词、格式、作者先验、韵脚、平仄和 word2vec 拼起来。

## 环境

进入项目目录：

```bash
cd C:\Users\liuqi\Desktop\机器学习概论\Poet
```

需要的库：

```bash
pip install requests beautifulsoup4 opencc-python-reimplemented pypinyin scikit-learn scipy numpy kmedoids gensim streamlit
```

数据、模型、结果都放在 `dataset/` 里。这个目录不进 git。根目录 `report.md` 也不进 git。

## 运行顺序

第一次完整跑：

```bash
python crawler.py
python label.py
python word2vec.py --limit 120000 --size 50 --epochs 5
python cluster.py --standard all --method all --limit 300
python cluster.py --standard all --method all --limit 300 --use_w2v
python classifier.py --standard all --train_limit 3000 --test_limit 1000 --use_w2v
```

如果只是试一下爬虫，可以少爬几个文件：

```bash
python crawler.py --max_file 2
```

启动界面：

```bash
streamlit run app.py
```

打开 Streamlit 给出的本地地址，一般是 `http://localhost:8501`。

## 文件说明

`.gitignore`

忽略爬下来的数据、中间结果、模型、图片、根目录 `report.md` 和 Python 缓存，避免把几百 MB 数据推到 git。

`classification_research.md`

分类标准调研。里面解释诗和词分别按什么划分，后面的 `label.py` 和 `config.py` 都按这份标准写。

`crawler.py`

爬全唐诗、全宋诗、全宋词，也保存平水韵、中华新韵、词牌字数表和每个字的平仄表。最后随机打乱，按 8:1:1 写成：

```text
dataset/poems/train.jsonl
dataset/poems/vaid.jsonl
dataset/poems/test.jsonl
```

每条数据大概有 `kind/title/cipai/author/paragraphs/content/tone/source`。

`label.py`

给每首诗词打弱标签。它不是人工真值，只是 LLM/人工规则辅助标签，后面分类器先把它当 true label 用。输出：

```text
dataset/labels/train_labels.jsonl
dataset/labels/vaid_labels.jsonl
dataset/labels/test_labels.jsonl
```

标签字段是：

```text
theme, season, festival, emotion, style, ci_style, form, rhyme, tone
```

没有明显证据就写 `N/A`。

`config.py`

特征文件。这里记录分类、聚类用的人工权重和特征生成方式。诗用题材、季节、节日、情感、诗风、体裁、韵脚、平仄、作者先验；词在这些之外，加词风、词牌、词牌字数和词牌句数。

`cluster.py`

聚类实验。直接调用：

```text
KMeans, KMedoids, PAM, AgglomerativeClustering, BisectingKMeans
```

例子：

```bash
python cluster.py --standard theme --method kmeans --limit 300
python cluster.py --standard ci_style --method all --limit 300 --use_w2v
```

结果写到：

```text
dataset/results/clusters/*.json
dataset/results/clusters/summary.md
```

JSON 里有 `acc/f1/silhouette/cluster_names/clusters`。`cluster_names` 是按簇里最多的真实弱标签给簇贴的名字。

`classifier.py`

分类实验。直接调用：

```text
KNN, weighted KNN, Naive Bayes, SVM, Decision Tree
Bagging_KNN, Bagging_WKNN, Bagging_Bayes, Bagging_SVM, Bagging_Tree
AdaBoost_Bayes, AdaBoost_SVM, AdaBoost_Tree
```

例子：

```bash
python classifier.py --standard all --train_limit 3000 --test_limit 1000
python classifier.py --standard all --train_limit 3000 --test_limit 1000 --use_w2v
```

它会训练、测试，然后把表写到根目录 `report.md`。这个文件被 gitignore 了。

`generator.py`

朴素生成器。诗按小句生成，词按词牌字数表拆成小句生成。模型主要是字级 bigram/马尔科夫，加一点 beam search 和韵脚限制。

生成诗：

```bash
python generator.py --kind poem --form 七言绝句 --theme 山水田园 --emotion 思乡怀人 --rhyme an
```

生成词：

```bash
python generator.py --kind ci --cipai 浣溪沙 --ci_style 婉约清丽 --emotion 离愁别绪 --word 月,柳
```

结果会打印在终端，也会写到：

```text
dataset/results/generate_sample.txt
```

`word2vec.py`

用所有诗词小句训练字级 word2vec。训练后模型在：

```text
dataset/models/word2vec.model
```

相似字例子写到：

```text
dataset/results/embedding_report.md
```

`app.py`

简单 Streamlit 界面，有三个页面：

- 分类：输入一首诗或词，返回标签和相似作品。
- 已有结果：看训练集标签分布和聚类摘要。
- 生成：按体裁、题材、情感、风格、词牌、韵脚、关键词生成诗词。

## 人工输入时填什么

在界面“分类”页：

- 如果输入诗，`词牌` 留空，只填 `文本`。
- 如果输入词，`词牌` 填词牌名，比如 `浣溪沙`，再填 `文本`。

返回的是一个 JSON：

```text
kind: tang_poem 或 song_ci
theme: 题材
season: 季节
festival: 节日
emotion: 情感
style: 诗风
ci_style: 词风，只有词会尽量判断
form: 诗体或小令/中调/长调
rhyme: 末字韵母
tone: 末字平仄
```

同页还会返回几首相似作品，包含 `score/title/author/kind/content`。

注意：当前界面人工输入走的是 `label.py` 的规则标签，不是 `classifier.py` 里 KNN/SVM/决策树那些离线实验模型。`classifier.py` 现在主要用于跑测试集指标。

## 分类标准

诗的分类标准：

- 题材：山水田园、边塞征战、送别留赠、羁旅思乡、咏史怀古、咏物言志、爱情闺怨、忧国民生、酬唱宴游、哲理咏怀、祝颂应制、悼亡祭奠。
- 时令节日：春、夏、秋、冬、节令；春节元日、元宵上元、寒食清明、端午、七夕、中秋、重阳、除夕岁暮。
- 情感：离愁别绪、思乡怀人、悲慨忧愤、闲适淡泊、豪迈昂扬、爱慕怨情、忧国伤时、喜悦赞美、孤寂清冷。
- 诗风：雄浑豪放、冲淡闲远、清新自然、绮丽纤秾、沉郁悲慨、高古典雅、含蓄委曲、飘逸旷达、劲健洗炼。
- 形式：五言绝句、七言绝句、五言律诗、七言律诗、古体长篇、杂言乐府、其他。
- 其他：韵脚、末字平仄、作者先验。

词的分类标准：

- 题材、时令、节日、情感也用上面的标准。
- 词风单独分：婉约清丽、豪放慷慨、清空骚雅、沉郁悲慨、典雅工丽、自然疏淡、俚俗谐趣。
- 形式分：小令、中调、长调。
- 词牌特征：词牌名、词牌常见字数、词牌句数。
- 其他：韵脚、末字平仄、作者先验。

## 当前结果文件

现在已有的主要结果：

```text
dataset/results/clusters/*.json
dataset/results/clusters/summary.md
dataset/results/embedding_report.md
dataset/results/generate_sample.txt
report.md
```

更详细的统计和实验结果见根目录 `report.md`，但它不会被 git 追踪。
