# 📂 Dataset Preparation Guide

The HAM10000 dataset is **not included** in this repository due to size and licensing restrictions.

This project uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset for skin lesion classification.

---

## 📥 Step 1 — Download the Dataset

Download the dataset from Kaggle:

🔗 https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

After downloading, extract the files.

Place the extracted files inside the `data/` directory like this:

data/
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
├── HAM10000_metadata.csv


---

## ⚙️ Step 2 — Organize the Dataset (Reproducible Split)

After placing the raw dataset inside the `data/` folder, simply run:

bash
python split_dataset.py


This script will:

Read HAM10000_metadata.csv

Perform a stratified 80:10:10 split (train/val/test)

Organize images into class-wise folders

Reproduce the exact dataset structure used during training

🔁 Reproducibility

The dataset split is fully reproducible because:

random.seed(42)


was used during the splitting process.

⚠️ Do not change the random seed if you want to reproduce the exact results reported in the paper.
