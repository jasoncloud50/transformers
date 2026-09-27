# GPT-Style Transformer 训练框架

基于 PyTorch 从零手写的 GPT-style Transformer 文本训练框架。

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.json` | 模型与训练配置（维度、层数、学习率、batch_size 等） |
| `tran.py` | 核心训练代码，包含多头注意力、残差连接、层归一化、正弦位置编码、学习率调度器等模块 |
| `prepare.py` | 数据预处理：使用 GPT-2 分词器将训练集与验证集文本转为 token 索引，生成 `train.bin` / `val.bin` / `meta.pkl` |
| `wikitext_train.txt` / `wikitext_val.txt` | WikiText 文本训练集与验证集 |
| `train.bin` / `val.bin` | 分词后的 token 索引数组（dtype: np.int32） |
| `meta.pkl` | 分词器子词多少 |
| `test.py` | 检测 CUDA 环境可用性 |

## 使用方法
python prepare.py   # 首次运行：分词并生成 .bin 等数据
python tran.py      # 开始训练

