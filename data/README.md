# 📂 Data folder

| Folder | What goes here | In Git? |
|---|---|---|
| `raw/` | `drugsComTrain_raw.tsv`, `drugsComTest_raw.tsv`, downloaded by `python main.py --step download` | ❌ no (112 MB, re-downloadable) |
| `processed/` | `train.parquet`, `val.parquet`, `test.parquet`, created by the prepare step | ❌ no (re-created) |

**Source:** [Drug Review Dataset (Drugs.com)](https://archive.ics.uci.edu/dataset/462), UCI Machine Learning Repository, CC BY 4.0.
Gräßer, F., Kallumadi, S., Malberg, H., & Zaunseder, S. (2018). *Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning.* Proceedings of the 2018 International Conference on Digital Health.
