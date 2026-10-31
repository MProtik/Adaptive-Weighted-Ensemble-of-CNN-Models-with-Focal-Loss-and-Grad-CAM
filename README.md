<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:1e3a5f,100:0ea5e9&height=200&section=header&text=Skin%20Lesion%20AI&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Adaptive%20Weighted%20Ensemble%20%E2%80%A2%20Focal%20Loss%20%E2%80%A2%20Grad-CAM&descAlignY=60&descSize=16&animation=fadeIn"/>

# Adaptive Weighted Ensemble of CNN Models with Focal Loss and Grad-CAM Explainability for Skin Lesion Classification

**Mokabbir Zaman Protik**

<p>
  <img src="https://img.shields.io/badge/Paper-Under%20Review%20%C2%B7%20Springer%20PRIA-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-HAM10000-6366f1?style=for-the-badge"/>
</p>

</div>

---

## Overview

Single CNN models for skin lesion classification often give unstable predictions and do not generalize well. Many ensembles try to fix this by averaging models equally, even though some models are more reliable than others. They also rarely show *why* they made a prediction.

This project combines three fine-tuned CNNs (**ResNet34, ResNet50, EfficientNet-B0**) into an ensemble where each model's vote is **weighted by its validation F1 score**. Models are trained with **Focal Loss** to handle HAM10000's heavy class imbalance, and **Grad-CAM** heatmaps show which part of the image each model looked at.

The paper is currently **under review at *Pattern Recognition and Image Analysis* (Springer), 2026**.

## Results

Test set results (custom stratified 80:10:10 split of HAM10000):

<div align="center">

| Model | Test Accuracy | Weighted F1 |
|:--|:--:|:--:|
| ResNet34 | 0.7587 | 0.7449 |
| ResNet50 | 0.7646 | 0.7477 |
| EfficientNet-B0 | 0.7478 | 0.7244 |
| **Weighted Ensemble (proposed)** | **0.7974** | **0.7782** |

</div>

The ensemble beats the best single model (ResNet50) by **+3.3 points in accuracy** and **+3.1 points in weighted F1**.

## Method

```
HAM10000 image
      │
      ▼
Preprocessing: resize 256×256 → random crop 224×224,
flips, ±5° rotation, brightness/contrast, ImageNet normalization
      │
      ├──────────────┬──────────────┐
      ▼              ▼              ▼
  ResNet34       ResNet50     EfficientNet-B0
  (w₁ = 0.33)    (w₂ = 0.34)   (w₃ = 0.33)
      │              │              │
      └──────── weighted soft voting ┘
                     │
                     ▼
        Final prediction (7 classes)
                     │
                     ▼
            Grad-CAM heatmaps
```

### 1. Dataset

HAM10000 has 10,015 dermoscopic images across 7 classes. It is very imbalanced: melanocytic nevi (NV) make up about 67% of the images.

| Class | Description | Images |
|:--:|:--|:--:|
| NV | Melanocytic nevi | 6,705 |
| MEL | Melanoma | 1,113 |
| BKL | Benign keratosis-like lesions | 1,099 |
| BCC | Basal cell carcinoma | 514 |
| AKIEC | Actinic keratosis / intraepithelial carcinoma | 327 |
| VASC | Vascular lesions | 142 |
| DF | Dermatofibroma | 115 |

The data was split **80% train / 10% validation / 10% test** with stratified sampling, so every split keeps the same class balance.

### 2. Handling class imbalance

Three things work together:
- **Focal Loss** instead of cross-entropy, with γ = 2 and α = 0.25. It gives less weight to easy, common examples and more to hard, rare ones.
- **Oversampling** of minority classes.
- **Class weights** adjusted for the imbalance.

### 3. Fine-tuning

All three models start from ImageNet weights. Only the final layer is replaced to output the 7 lesion classes, and every model uses the same training setup so the comparison is fair.

| Setting | Value |
|:--|:--|
| Optimizer | Adam, learning rate 1×10⁻⁴, weight decay 3×10⁻⁶ |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 3, on validation F1) |
| Batch size | 64 |
| Epochs | 30 |
| Early stopping | patience 5 (never triggered; all models trained smoothly for 30 epochs) |
| Checkpoint | best validation F1 |
| Framework | PyTorch, GPU |

### 4. Adaptive weighted ensemble

Each model's weight comes from its validation F1 score, normalized so the weights add up to 1:

$$w_m = \frac{F1_m}{\sum_{i=1}^{3} F1_i}$$

The final probability for each class is the weighted sum of the three models' probabilities, and the class with the highest score wins:

$$y_i = w_1 p^{(1)}_i + w_2 p^{(2)}_i + w_3 p^{(3)}_i, \qquad \hat{y} = \arg\max_i \, y_i$$

The resulting weights were **w₁ = 0.33 (ResNet34), w₂ = 0.34 (ResNet50), w₃ = 0.33 (EfficientNet-B0)**. They are close because the three models performed similarly, but the stronger model still gets more say.

### 5. Grad-CAM explainability

Grad-CAM heatmaps were generated for each model and for the ensemble on three example lesions:

| Lesion | Image ID | What happened |
|:--|:--|:--|
| VASC | ISIC_0033450 | All models found the central vascular area. The ensemble kept that focus and covered the lesion edge slightly better. |
| NV | ISIC_0030957 | All models found the pigmented area. The ensemble heatmap was balanced and followed the lesion outline closely. |
| DF | ISIC_0029967 | EfficientNet-B0 showed almost no activation on the lesion. The ensemble still focused correctly, using the ResNet models' attention. |

The DF case shows the main benefit: when one model fails, the ensemble can still look at the right place.

## Repository structure

```
├── Dataset/      HAM10000 setup and preprocessing
├── Notebooks/    Training, ensemble, evaluation and Grad-CAM
├── Results/      Training curves, metrics and heatmaps
└── README.md
```

## How to run

```bash
git clone https://github.com/MProtik/Adaptive-Weighted-Ensemble-of-CNN-Models-with-Focal-Loss-and-Grad-CAM.git
cd Adaptive-Weighted-Ensemble-of-CNN-Models-with-Focal-Loss-and-Grad-CAM
pip install torch torchvision grad-cam numpy pandas matplotlib scikit-learn jupyter
```

1. Download HAM10000 from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) or the [ISIC archive](https://challenge.isic-archive.com/data/) and place it in `Dataset/`.
2. Open the notebooks in `Notebooks/` and run them in order: train the three base models, build the weighted ensemble, then generate Grad-CAM heatmaps.
3. Outputs are saved in `Results/`.

## Limitations

- Rare classes (DF, VASC) still have weaker recall because there are so few examples.
- Tested on HAM10000 only, so performance on images from other clinics or cameras is unknown.
- Grad-CAM is a visual, qualitative check. It does not measure how reliable an explanation is.
- Running three models costs more compute than one, which matters for real-time or edge use.

## Future work

- Add more varied architectures such as ConvNeXt, DenseNet and Vision Transformers.
- Try Bayesian or uncertainty-aware weighting so predictions come with confidence estimates.
- Use Grad-CAM++ or Layer-wise Relevance Propagation for explanations that can be measured.

## Citation

```bibtex
@unpublished{protik2026ensemble,
  title  = {Adaptive Weighted Ensemble of CNN Models with Focal Loss and
            Grad-CAM Explainability for Skin Lesion Classification},
  author = {Protik, Mokabbir Zaman},
  year   = {2026},
  note   = {Under review at Pattern Recognition and Image Analysis (Springer)}
}
```

## Key references

1. He et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.
2. Tan & Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.
3. Lin et al. *Focal Loss for Dense Object Detection.* ICCV 2017.
4. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
5. Tschandl et al. *The HAM10000 Dataset.* Scientific Data 2018.
6. Li & Luo. *Performance-weighted-voting model.* Quantitative Biology 2020.

## Author

**Mokabbir Zaman Protik**
B.Sc. in Computer Science, The People's University of Bangladesh

[![GitHub](https://img.shields.io/badge/GitHub-MProtik-181717?style=flat-square&logo=github)](https://github.com/MProtik)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:mokabbir.zaman.protik@gmail.com)
