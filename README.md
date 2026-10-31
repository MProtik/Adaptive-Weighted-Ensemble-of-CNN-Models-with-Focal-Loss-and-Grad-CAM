<div align="center">

<!-- ANIMATED HEADER BANNER -->

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:1e3a5f,100:0ea5e9&height=200&section=header&text=Skin%20Lesion%20AI&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Adaptive%20Weighted%20Ensemble%20%E2%80%A2%20HAM10000%20%E2%80%A2%20Grad-CAM%20Explainability&descAlignY=60&descSize=16&animation=fadeIn"/>

<!-- BADGES ROW 1 - Status & Core Info -->

<p>
  <img src="https://img.shields.io/badge/Paper-Under%20Review%20%C2%B7%20Springer%20PRIA-22c55e?style=for-the-badge&logo=springer&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-Framework-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge&logo=opensourceinitiative&logoColor=white"/>
</p>

<!-- BADGES ROW 2 - Models & Dataset -->

<p>
  <img src="https://img.shields.io/badge/Dataset-HAM10000-6366f1?style=for-the-badge&logo=kaggle&logoColor=white"/>
  <img src="https://img.shields.io/badge/ResNet34-Model%201-0ea5e9?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ResNet50-Model%202-06b6d4?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/EfficientNet--B0-Model%203-8b5cf6?style=for-the-badge"/>
</p>

<!-- BADGES ROW 3 - Techniques -->

<p>
  <img src="https://img.shields.io/badge/Focal%20Loss-Class%20Imbalance-ef4444?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Grad--CAM-Explainability-f97316?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Transfer%20Learning-ImageNet%20Pretrained-10b981?style=for-the-badge"/>
</p>

</div>

-----

## 📌 At a Glance

> **A weighted ensemble transfer learning framework** that fuses ResNet34, ResNet50, and EfficientNet-B0 for automated skin cancer classification on the HAM10000 dataset — featuring adaptive model weighting, Focal Loss for class imbalance, and Grad-CAM heatmaps for clinical explainability.

<div align="center">

|Metric           |ResNet34|ResNet50|EfficientNet-B0|🏆 Weighted Ensemble|
|:---------------:|:------:|:------:|:-------------:|:-----------------:|
|**Test Accuracy**|0.7587  |0.7646  |0.7478         |**0.7974**         |
|**Weighted F1**  |0.7449  |0.7477  |0.7244         |**0.7782**         |


> ✅ The ensemble outperforms every individual model — **+3.3 points accuracy and +3.1 points weighted F1** over the best single model (ResNet50).

</div>

-----

## 🧠 Why This Research Matters

Skin cancer is one of the most prevalent cancers globally. Early diagnosis dramatically improves survival rates, yet traditional methods are:

- 🕒 **Time-consuming** — dependent on specialist availability
- 👁️ **Subjective** — even experienced dermatologists misclassify lesions
- ⚖️ **Inconsistent** — no standardized automated pipeline

This project builds an **explainable AI pipeline** that addresses all three.

-----

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  HAM10000 Input Image                    │
│              (10,015 dermoscopic images)                 │
└─────────────────────┬───────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   Preprocessing Pipeline │
          │  • Resize → 224×224     │
          │  • ImageNet Normalization│
          │  • Augmentation         │
          │  • Class Rebalancing    │
          └────────────┬────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌──────────┐  ┌──────────┐  ┌──────────────┐
  │ ResNet34 │  │ ResNet50 │  │EfficientNet  │
  │ 34 layers│  │ 50 layers│  │     B0       │
  │ w₁= 0.33 │  │ w₂= 0.34 │  │  w₃= 0.33   │
  └────┬─────┘  └────┬─────┘  └──────┬───────┘
       │              │               │
       └──────────────▼───────────────┘
                       │
          ┌────────────▼────────────┐
          │  Adaptive Soft Voting   │
          │  ŷ = w₁p₁ + w₂p₂ + w₃p₃│
          │  (F1-normalized weights) │
          └────────────┬────────────┘
                       │
        ┌──────────────▼──────────────┐
        │     Final Prediction         │
        │  7 Skin Lesion Classes       │
        └──────────────┬──────────────┘
                       │
          ┌────────────▼────────────┐
          │   Grad-CAM Heatmaps     │
          │ (Visual Explainability) │
          └─────────────────────────┘
```

-----

## 📂 Repository Structure

```
📦 Adaptive-Weighted-Ensemble-of-CNN-Models-with-Focal-Loss-and-Grad-CAM/
├── 📁 Dataset/      # HAM10000 setup and preprocessing
├── 📁 Notebooks/    # Training, ensemble, evaluation and Grad-CAM notebooks
├── 📁 Results/      # Training curves, metrics and Grad-CAM heatmaps
└── 📄 README.md
```

-----

## ⚙️ Methodology

### 1️⃣ Dataset — HAM10000

<div align="center">

|Class  |Description          |Images|% of Dataset|
|:-----:|:--------------------|:----:|:----------:|
|`NV`   |Melanocytic nevi     |6,705 |66.9%       |
|`MEL`  |Melanoma             |1,113 |11.1%       |
|`BKL`  |Benign keratosis-like|1,099 |10.9%       |
|`BCC`  |Basal cell carcinoma |514   |5.1%        |
|`AKIEC`|Actinic keratosis    |327   |3.2%        |
|`VASC` |Vascular lesions     |142   |1.4%        |
|`DF`   |Dermatofibroma       |115   |1.1%        |

</div>


> ⚠️ **Class Imbalance Challenge**: NV makes up ~67% of the dataset — a major challenge addressed by Focal Loss + oversampling.

### 2️⃣ Focal Loss for Class Imbalance

Instead of standard Cross-Entropy, we use **Focal Loss** (Lin et al., 2017):

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

With canonical values **γ = 2, α = 0.25** — this down-weights easy majority-class examples and focuses training on hard, rare cases like DF and VASC.

### 3️⃣ Adaptive Weighted Ensemble

Ensemble weights are derived from each model’s **validation F1 score**, then normalized:

$$w_m = \frac{F1_m}{\sum_{i=1}^{3} F1_i}$$

Final prediction via soft voting:

$$\hat{y} = \arg\max \left( w_1 p^{(1)}_i + w_2 p^{(2)}_i + w_3 p^{(3)}_i \right)$$

This ensures **stronger models have greater influence** — unlike naive averaging which treats all models equally.

### 4️⃣ Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017) generates **visual heatmaps** showing which image regions drove the model’s decision — critical for **clinical trust**.

Key finding: When EfficientNet-B0 failed to activate on a DF lesion, the ensemble’s heatmap **recovered correct focus** using the ResNet models’ attention.

-----

## 🚀 Quickstart

### Prerequisites

```bash
pip install torch torchvision efficientnet-pytorch grad-cam numpy matplotlib scikit-learn pandas jupyter
```

### Clone

```bash
git clone https://github.com/MProtik/Adaptive-Weighted-Ensemble-of-CNN-Models-with-Focal-Loss-and-Grad-CAM.git
cd Adaptive-Weighted-Ensemble-of-CNN-Models-with-Focal-Loss-and-Grad-CAM
```

### Download HAM10000

```bash
kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection
unzip skin-lesion-analysis-toward-melanoma-detection.zip -d Dataset/
```

### Run

Open the notebooks in `Notebooks/` and run them in order: train the three base models, build the weighted ensemble, then generate Grad-CAM heatmaps. Outputs are saved to `Results/`.

-----

## 📊 Training Configuration

|Hyperparameter|Value                                            |
|:-------------|:------------------------------------------------|
|Optimizer     |Adam                                             |
|Learning Rate |`1 × 10⁻⁴`                                       |
|Weight Decay  |`3 × 10⁻⁶`                                       |
|Batch Size    |64                                               |
|Epochs        |30                                               |
|LR Scheduler  |ReduceLROnPlateau (factor=0.5, patience=3)       |
|Early Stopping|patience=5 (did not trigger — stable convergence)|
|Loss Function |Focal Loss (γ=2, α=0.25)                         |
|Input Size    |224 × 224                                        |
|Pretrained    |ImageNet                                         |
|Split         |80% train / 10% val / 10% test                   |

-----

## 📈 Results Summary

<div align="center">

```
Individual Models vs. Weighted Ensemble
────────────────────────────────────────────────────────
Model             Test Accuracy    Weighted F1
────────────────────────────────────────────────────────
ResNet34          0.7587           0.7449
ResNet50          0.7646           0.7477
EfficientNet-B0   0.7478           0.7244
────────────────────────────────────────────────────────
Weighted Ensemble 0.7974 ▲+3.3 pts 0.7782 ▲+3.1 pts
────────────────────────────────────────────────────────
```

</div>

### Key Takeaways

- ✅ **Ensemble beats all individual models** on both accuracy and F1
- ✅ **Focal Loss** improved recall for underrepresented classes (DF, VASC)
- ✅ **Grad-CAM ensemble heatmaps** are more stable than any single model’s — even compensating when one model fails
- ✅ **No overfitting** observed — training and validation curves converge cleanly across all 30 epochs

-----

## 🔬 Grad-CAM Visual Results

> Heatmaps generated for three representative lesion cases: VASC, NV, and DF

|Case               |Patient   |Lesion         |Ensemble Behavior                                              |
|:-----------------:|:--------:|:-------------:|:--------------------------------------------------------------|
|VASC (ISIC_0033450)|Male, 15  |Chest          |All models correctly focused on central vascular region        |
|NV (ISIC_0030957)  |Female, 35|Back           |Ensemble produced sharp, balanced pigmented-area coverage      |
|DF (ISIC_0029967)  |Female, 65|Lower extremity|EfficientNet-B0 failed; ensemble recovered via ResNet attention|


> 📁 Heatmap outputs are in the `Results/` folder.

-----

## 🧩 Technical Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Kaggle](https://img.shields.io/badge/HAM10000-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?style=flat-square&logo=nvidia&logoColor=white)

</div>

-----

## 📖 Citation

If you use this work, please cite:

```bibtex
@unpublished{protik2026ensemble,
  title  = {Adaptive Weighted Ensemble of CNN Models with Focal Loss and
            Grad-CAM Explainability for Skin Lesion Classification},
  author = {Protik, Mokabbir Zaman},
  year   = {2026},
  note   = {Under review at Pattern Recognition and Image Analysis (Springer)}
}
```

-----

## 📚 References

|#  |Paper                                                                           |
|:--|:-------------------------------------------------------------------------------|
|[1]|He et al. — *Deep Residual Learning for Image Recognition*, CVPR 2016           |
|[2]|Tschandl et al. — *The HAM10000 Dataset*, Scientific Data 2018                  |
|[3]|Tan & Le — *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019         |
|[4]|Selvaraju et al. — *Grad-CAM: Visual Explanations from Deep Networks*, ICCV 2017|
|[5]|Lin et al. — *Focal Loss for Dense Object Detection*, ICCV 2017                 |

-----

## 🚧 Limitations & Future Work

**Current Limitations:**

- Tested on HAM10000 only — generalization to other skin image datasets is untested
- Rare classes (DF, VASC) still underperform due to extreme data scarcity
- Grad-CAM provides qualitative, not quantitative, explanation reliability
- Ensemble inference is slower than single-model deployment

**Planned Extensions:**

- [ ] Add ConvNeXt, DenseNet, and Vision Transformers (ViT) to the ensemble
- [ ] Implement Bayesian uncertainty-aware weighting for calibrated confidence
- [ ] Upgrade to Grad-CAM++ / Layer-wise Relevance Propagation (LRP)
- [ ] GAN-based augmentation for minority class oversampling

-----

## 👤 Author

<div align="center">

**Mokabbir Zaman Protik**  
Department of Computer Science and Engineering  
The People’s University of Bangladesh, Narsingdi

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/MProtik)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/Mokabbir_Protik)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:Mokabbir.Zaman.Protik@gmail.com)

</div>

-----

<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0ea5e9,50:1e3a5f,100:0f172a&height=120&section=footer"/>

*Built for clinical impact — where AI meets dermatology.*

</div>
