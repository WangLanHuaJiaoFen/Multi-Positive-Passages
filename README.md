<h1 align="center">😊 Training Dense Retrievers with Multiple Positive Passages
<h5 align="center"> If you like our work, 

please give us a star ⭐ on GitHub</h5>
<div align="center"> 

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2602.12727)
[![Github](https://img.shields.io/badge/github-repo-blue?logo=github
)](https://github.com/WangLanHuaJiaoFen/Multi-Positive-Passages)
</div>

## 🔈News
[🎉 2026-05-18] We are excited to release the code for our paper.

[🎉 2026-05-16] Our paper "Training Dense Retrievers with Multiple Positive Passages" has been accepted at **KDD2026 Cycle 2** for the Research Track!

## Overview
This repository contains the code used in our paper: "Training Dense Retrievers with Multiple Positive Passages". 

Datasets is coming soon...

Our paper presents a systematic study of multi-positive optimization objectives for dense retrieval, unifying representative listwise and pairwise losses under a contrastive learning framework. Through theoretical analysis and extensive experiments on NQ, MS MARCO, and BEIR, we investigate how different objectives leverage multiple positive passages and demonstrate that LSEPair achieves strong robustness and retrieval performance across diverse supervision settings.

## 🚀Quick Start

Based on the v1 branch of the [Tevatron](https://github.com/texttron/tevatron/tree/tevatron-v1) ,the implementation of multi-positive loss functions is provided in [src/tevatron/modeling/encoder.py](https://github.com/WangLanHuaJiaoFen/Multi-Positive-Passages/blob/main/src/tevatron/modeling/encoder.py).

### 1. Installation
```bash
git clone https://github.com/texttron/tevatron.git

cd Multi-Positive-Passages/

conda create -n multi-pos python=3.9
conda activate multi-pos

pip install -e .
```

### 2. Training and Evaluating on MS-MARCO
See [examples/]().

The examples is coming soon...

## 📧Contact
For any questions or feedback, please reach out to us @ wangbenben@stu.xidian.edu.cn

## Citation
coming soon...

## 📜 License
This project is released under the [Apache-2.0 License](http://www.apache.org/licenses/LICENSE-2.0).