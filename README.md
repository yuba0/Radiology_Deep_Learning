# RadIA-DL — Système d'aide au tri radiologique

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)

**Système de deep learning pour la prédiction de pathologies thoraciques à partir de radiographies.**

*Classification supervisée · Détection d'anomalies · Multimodalité image+texte · Démonstrateur interactif*

</div>

---

## 📋 Table des matières

1. [Contexte et objectifs](#-contexte-et-objectifs)
2. [Datasets](#-datasets)
3. [Architecture du projet](#-architecture-du-projet)
4. [Installation](#-installation)
5. [Lancement](#-lancement)
6. [Composantes du projet](#-composantes-du-projet)
   - [Classification supervisée](#1-classification-supervisée)
   - [Détection d'anomalies](#2-détection-danomalies--autoencoder)
   - [Multimodalité](#3-multimodalité-image--texte)
   - [MLflow](#4-tracking-expérimental--mlflow)
   - [Démonstrateur Flask](#5-démonstrateur-flask)
7. [Résultats](#-résultats)
8. [Captures d'écran](#-captures-décran)
9. [Configuration matérielle](#-configuration-matérielle)
10. [Équipe](#-équipe)

---

## 🎯 Contexte et objectifs

Ce projet s'inscrit dans le cadre d'un projet de Deep Learning appliqué à l'imagerie médicale. L'objectif est de concevoir un **système d'aide au tri radiologique** capable de :

- **Prédire des pathologies thoraciques** à partir de radiographies via des architectures profondes
- **Identifier des cas atypiques** ou hors distribution grâce à un modèle de détection d'anomalies
- **Exploiter le contexte textuel** (annotations de radiologues) via une approche multimodale
- **Rendre le système accessible** via un démonstrateur applicatif interactif

Le projet couvre l'ensemble du pipeline machine learning : de l'analyse exploratoire des données à la mise en production d'un prototype fonctionnel, en passant par la modélisation supervisée, la détection d'anomalies non supervisée et la fusion multimodale.

---

## 📊 Datasets

### Dataset principal — ChestMNIST (Classification supervisée + AE)

| Propriété | Valeur |
|-----------|--------|
| Source | MedMNIST v2 / NIH Chest X-Ray |
| Résolution | 64 × 64 pixels (niveaux de gris) |
| Taille train | 78 468 images |
| Taille val | 11 219 images |
| Taille test | 22 433 images |
| Nombre de classes | 14 pathologies (multi-label) |
| Format | `.npz` prétraité (~200 Mo) |

**Les 14 pathologies :** Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia

**Déséquilibre observé :**
- Classe majoritaire : Infiltration → 13 914 images
- Classe minoritaire : Hernia → 144 images
- Ratio déséquilibre : ~97×

### Dataset multimodal — NIH Chest X-Ray (Complet)

| Propriété | Valeur |
|-----------|--------|
| Source | NIH Clinical Center |
| Images totales | 112 120 radiographies |
| Résolution originale | ~1024 × 1024 pixels |
| Résolution utilisée | 64 × 64 pixels (redimensionnées) |
| Annotations | Labels cliniques officiels NIH |
| Taille brute | ~45 Go |
| Train/Val/Test | 70% / 15% / 15% |
| Plateforme | Kaggle (dataset nih-chest-xrays/data) |

---

## 🗂️ Architecture du projet

```
RadIA-DL/
├── app.py                          # Backend Flask — API + chargement modèles
├── requirements.txt                # Dépendances Python
├── README.md                       # Documentation
│
├── templates/
│   └── index.html                  # Interface web complète (dark mode médical)
│
├── Modèles/
│   ├── CNN_scratch_best.pth        # CNN entraîné depuis zéro
│   ├── ResNet50_finetune_best.pth  # ResNet50 fine-tuné
│   ├── ViT_small_best.pth          # Vision Transformer Small
│   ├── autoencoder_best.pth        # Autoencoder convolutionnel
│   └── Multimodèle/
│       ├── MM_ImageOnly_best.pth   # Modèle image seule (NIH)
│       ├── MM_TextOnly_best.pth    # Modèle texte seul (NIH)
│       └── MM_Multimodal_best.pth  # Modèle multimodal fusionné (NIH)
│
└── Notebook/
    ├── projet_radiologie.ipynb     # Notebook principal (Colab) — phases 1 à 11
    └── modele-multi.ipynb          # Notebook multimodal (Kaggle) — NIH complet
```

---

## ⚙️ Installation

### Prérequis

- Python 3.10+
- pip
- (Optionnel) GPU CUDA pour l'inférence rapide

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/ton-username/RadIA-DL.git
cd RadIA-DL

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Contenu de `requirements.txt`

```
flask==3.0.0
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.24.0
Pillow>=10.0.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
```

---

## 🚀 Lancement

```bash
python app.py
```

Ouvre ensuite **http://localhost:5000** dans ton navigateur.

Le système charge automatiquement les 7 modèles au démarrage. Le message `✅ Tous les modèles chargés` confirme que tout est opérationnel.

---

## 🧠 Composantes du projet

### 1. Classification supervisée

Trois architectures profondes ont été comparées sur ChestMNIST 64px, en résolvant une tâche de **classification multi-label** (une image peut avoir plusieurs pathologies simultanément).

#### Modèle 1 — CNN from scratch

Architecture CNN construite et entraînée entièrement depuis zéro, servant de **baseline** pour les comparaisons.

```
Input (1×64×64)
    ↓
Conv Block 1 : Conv2d(1→32) + BN + ReLU × 2 + MaxPool + Dropout(0.25)
    ↓
Conv Block 2 : Conv2d(32→64) + BN + ReLU × 2 + MaxPool + Dropout(0.25)
    ↓
Conv Block 3 : Conv2d(64→128) + BN + ReLU × 2 + MaxPool + Dropout(0.25)
    ↓
AdaptiveAvgPool → Flatten
    ↓
FC(128→256) + ReLU + Dropout(0.5)
    ↓
FC(256→14) → Sigmoid
```

| Paramètre | Valeur |
|-----------|--------|
| Paramètres totaux | 323 950 |
| Optimiseur | Adam (lr=1e-3, wd=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=3) |
| Epochs | 20 |
| Early stopping | patience=8 |

#### Modèle 2 — ResNet50 Transfer Learning

Fine-tuning en deux phases d'un ResNet50 pré-entraîné sur ImageNet, avec adaptation du premier `conv1` pour les images en niveaux de gris (1 canal).

**Phase 1** — Entraînement de la tête uniquement (backbone gelé sauf layer4) :
- lr = 1e-3, epochs = 15

**Phase 2** — Fine-tuning complet (tous les paramètres dégelés) :
- lr = 5e-5, epochs = 10, CosineAnnealingLR

| Paramètre | Valeur |
|-----------|--------|
| Paramètres entraînables (phase 1) | 16 021 006 |
| Architecture tête | Dropout(0.5) → FC(2048→512) → ReLU → Dropout(0.3) → FC(512→14) |

#### Modèle 3 — Vision Transformer (ViT Small)

ViT Small patch16/224 pré-entraîné via `timm`, avec les 2 derniers blocs d'attention dégelés. Les images 64px sont interpolées à 224px pour correspondre à l'entrée du modèle.

```
Input (1×64×64)
    ↓
Interpolation bilinéaire → (1×224×224)
    ↓
ViT Small patch16 (backbone gelé sauf blocks 10, 11, norm)
    ↓
LayerNorm + Dropout(0.4)
    ↓
FC(384→256) + GELU + Dropout(0.3)
    ↓
FC(256→14) → Sigmoid
```

| Paramètre | Valeur |
|-----------|--------|
| Paramètres entraînables | 3 667 982 |
| Optimiseur | AdamW (lr=5e-5, wd=0.05) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10) |

#### Gestion du déséquilibre

Deux mécanismes combinés pour gérer le ratio 97× entre classes :

1. **WeightedRandomSampler** — surreprésente les classes rares lors de l'échantillonnage
2. **BCEWithLogitsLoss avec pos_weight** — pénalise davantage les faux négatifs sur les classes rares

```python
pos_weight = (N_total - class_freq) / (class_freq + 1)
# Hernia : pos_weight ≈ 400
# Infiltration : pos_weight ≈ 5
```

---

### 2. Détection d'anomalies — Autoencoder

Un **autoencoder convolutionnel** est entraîné à reconstruire les radiographies. Le score d'anomalie est l'erreur de reconstruction MSE — une erreur élevée indique une image atypique ou hors distribution.

#### Architecture

```
ENCODEUR
Input (1×64×64)
    → Conv(1→32, stride=2)  → ReLU  → (32×32×32)
    → Conv(32→64, stride=2) → ReLU  → (64×16×16)
    → Conv(64→128, stride=2)→ ReLU  → (128×8×8)
    → Conv(128→64, stride=2)→ ReLU  → (64×4×4)  [espace latent]

DÉCODEUR
    → ConvT(64→128, stride=2) → ReLU → (128×8×8)
    → ConvT(128→64, stride=2) → ReLU → (64×16×16)
    → ConvT(64→32, stride=2)  → ReLU → (32×32×32)
    → ConvT(32→1, stride=2)   → Tanh → (1×64×64)
```

| Paramètre | Valeur |
|-----------|--------|
| Paramètres totaux | 332 865 |
| Fonction de perte | MSELoss |
| Optimiseur | Adam (lr=1e-3) |
| Epochs | 20 |
| Loss finale | 0.001899 |

#### Résultats

| Métrique | Valeur |
|----------|--------|
| Score min (cas normal) | 0.0001 |
| Score max (cas atypique) | 0.0064 |
| Score moyen | 0.0013 |
| Seuil 95e percentile | ~0.0035 |

Les cas à fort score d'anomalie présentent des anomalies visuelles objectives (orientation inhabituelle, densité anormale, artefacts), validant l'approche non supervisée.

---

### 3. Multimodalité image + texte

Trois modèles ont été entraînés sur le **dataset NIH complet** (112 120 images) avec leurs annotations cliniques officielles, pour comparer les apports respectifs de l'image et du texte.

#### Stratégie de fusion — Late Fusion

La stratégie de **fusion tardive** a été choisie : les représentations image et texte sont calculées indépendamment puis concaténées avant la couche de décision. Cette approche est plus robuste à l'absence d'une modalité.

```
Branche image                    Branche texte
CNN(1→128) → proj(128→256)      Embedding(19×64) → BiLSTM(128) → proj(256→256)
                    ↓                                    ↓
               Concaténation (512)
                    ↓
             FC(512→256) → ReLU → Dropout(0.4) → FC(256→14)
```

#### Vocabulaire NIH

Le vocabulaire est extrait directement des labels cliniques officiels NIH (19 mots uniques correspondant aux 14 pathologies + "No Finding"). Ces labels sont les annotations posées par les radiologues du NIH Clinical Center.

#### Résultats comparatifs (Val Set)

| Modèle | ROC-AUC | Observation |
|--------|---------|-------------|
| Image seule | 0.6877 | Apprentissage visuel pur |
| Texte seul | 1.0000 | Data leakage (labels = cibles) |
| Multimodal | 0.9997 | Dominé par le texte |

#### Analyse de robustesse

| Condition | ROC-AUC |
|-----------|---------|
| Avec texte | 0.9998 |
| Sans texte (zeros) | 0.5001 |
| Dégradation | 50.0% |

**Note importante** : Le score parfait du modèle texte seul s'explique par un **data leakage structurel** — les annotations NIH sont des labels cliniques directement corrélés aux cibles de prédiction. Dans un contexte réel, le texte serait un compte-rendu narratif indépendant (ex: MIMIC-CXR). Ce résultat illustre l'importance d'une séparation stricte features/labels en multimodalité.

---

### 4. Tracking expérimental — MLflow

Toutes les expériences sont tracées avec **MLflow**, stocké sur Google Drive pour la persistance entre les sessions Colab.

#### Organisation des runs

| Run | Modèle | Best ROC-AUC val |
|-----|--------|-----------------|
| CNN_scratch | CNN from scratch | 0.7091 |
| ResNet50_TL | ResNet50 phase 1 | 0.6882 |
| ResNet50_finetune | ResNet50 fine-tune | 0.7123 |
| ViT_small | ViT Small | 0.7286 |
| ConvAutoencoder | Autoencoder | — (MSE) |
| MM_ImageOnly | Image seule NIH | 0.6877 |
| MM_TextOnly | Texte seul NIH | 1.0000 |
| MM_Multimodal | Multimodal NIH | 0.9997 |

#### Métriques loguées par run

- Hyperparamètres : modèle, epochs, batch_size, lr, img_size, seed
- Métriques par epoch : train_loss, val_loss, val_roc_auc, val_f1_macro, val_pr_auc
- Artefacts : courbes d'apprentissage (.png), meilleur modèle (.pth)
- Métrique finale : best_val_roc_auc

#### Lancement de l'interface MLflow

```bash
mlflow ui --backend-store-uri file:///chemin/vers/mlruns
```

---

### 5. Démonstrateur Flask

Interface web professionnelle développée avec **Flask** (backend) et **HTML/CSS/JS vanilla** (frontend). Design dark mode médical inspiré des systèmes PACS (Picture Archiving and Communication Systems).

#### Fonctionnalités

**Onglet 1 — Analyse supervisée :**
- Upload d'une radiographie (drag & drop ou clic)
- Prédiction simultanée via CNN, ResNet50 et ViT
- Affichage des top-5 pathologies par modèle avec barres de confiance
- Score d'anomalie via autoencoder avec jauge visuelle
- Visualisation Grad-CAM des zones d'activation importantes
- Reconstruction de l'image par l'autoencoder

**Onglet 2 — Analyse multimodale :**
- Upload radiographie + saisie des annotations cliniques
- Comparaison côte à côte : image seule / texte seul / multimodal fusionné
- Affichage des top-5 prédictions par modalité

#### Architecture technique

```
Frontend (HTML/CSS/JS)
    ↕ Fetch API (JSON)
Backend Flask (app.py)
    ├── /predict      → CNN + ResNet50 + ViT + AE + GradCAM
    └── /multimodal   → ImageOnly + TextOnly + Multimodal
```

#### Endpoints API

| Endpoint | Méthode | Input | Output |
|----------|---------|-------|--------|
| `/` | GET | — | Interface HTML |
| `/predict` | POST | image (multipart) | prédictions JSON + images base64 |
| `/multimodal` | POST | image + texte | comparaison 3 modèles JSON |

---

## 📈 Résultats

### Classification supervisée — Test Set

| Modèle | ROC-AUC | F1 Macro | PR-AUC |
|--------|---------|----------|--------|
| CNN scratch | 0.7046 | 0.1019 | 0.1187 |
| ResNet50 TL | 0.7154 | 0.1146 | 0.1278 |
| **ViT Small** | **0.7190** | **0.1140** | **0.1331** |

**Meilleur modèle : ViT Small** avec ROC-AUC = 0.7190 sur le test set.

### Analyse critique des résultats

**Pourquoi le F1 est bas ?**
Le F1 macro faible (~0.10) est attendu dans ce contexte : avec 14 classes fortement déséquilibrées et un threshold fixe à 0.5, le modèle prédit très peu de positifs. Ce résultat s'améliore significativement avec un threshold optimisé par classe.

**Pourquoi ResNet50 ne surpasse pas le CNN scratch ?**
La réinitialisation du premier `conv1` (pour adapter 3 canaux → 1 canal) détruit une partie des features ImageNet appris. Le transfer learning est moins efficace sur des images médicales en niveaux de gris que sur des images naturelles RGB.

**Pourquoi ViT est le meilleur ?**
Le mécanisme d'attention globale du ViT capture mieux les relations spatiales longue distance dans les radiographies (ex: relation entre la taille du cœur et les poumons pour détecter la cardiomégalie).

### Détection d'anomalies

| Métrique | Valeur |
|----------|--------|
| Loss reconstruction finale | 0.001899 |
| Score anomalie moyen | 0.0013 |
| Score max (atypique) | 0.0064 |
| Réduction loss (epoch 1→20) | 90.5% |

### Multimodalité NIH

| Modèle | ROC-AUC val | ROC-AUC sans texte |
|--------|-------------|-------------------|
| Image seule | 0.6877 | — |
| Texte seul | 1.0000 | — |
| Multimodal | 0.9997 | 0.5001 |

---

## 📸 Captures d'écran

> *Ajouter ici les captures d'écran du démonstrateur*

### Interface principale — Analyse supervisée
![Interface supervisée](screenshots/interface_supervisee.png)

### Visualisation Grad-CAM
![GradCAM](screenshots/gradcam.png)

### Score d'anomalie
![Anomalie](screenshots/anomalie.png)

### Interface multimodale
![Multimodal](screenshots/multimodal.png)

---

## 🖥️ Configuration matérielle

| Composant | Valeur |
|-----------|--------|
| Plateforme entraînement | Google Colab (CNN/ResNet/ViT/AE) + Kaggle (Multimodal NIH) |
| GPU entraînement | NVIDIA Tesla T4 (16 Go VRAM) |
| Plateforme inférence | CPU local (démonstrateur Flask) |
| Résolution images | 64×64 pixels (ChestMNIST) / 64×64 pixels (NIH redimensionné) |

### Temps d'entraînement estimés (T4)

| Modèle | Epochs | Temps estimé |
|--------|--------|-------------|
| CNN scratch | 20 | ~25 min |
| ResNet50 phase 1 | 15 | ~20 min |
| ResNet50 fine-tune | 10 | ~15 min |
| ViT Small | 16 | ~2h30 |
| Autoencoder | 20 | ~20 min |
| MM ImageOnly | 5 | ~1h |
| MM TextOnly | 5 | ~1h |
| MM Multimodal | 5 | ~1h30 |

### Seed de reproductibilité

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
```

---

## 🔧 Choix techniques

### Frameworks

| Usage | Framework |
|-------|-----------|
| Deep Learning | PyTorch 2.0+ |
| Vision Transformer | timm |
| Visualisation | Matplotlib, Seaborn |
| Métriques | scikit-learn |
| Tracking | MLflow |
| Démonstrateur | Flask 3.0 |
| Grad-CAM | OpenCV + PyTorch hooks |

### Régularisation

- **Dropout** : 0.25 (conv), 0.3-0.5 (FC)
- **Batch Normalization** : après chaque conv
- **Weight Decay** : 1e-4 (Adam), 0.05 (AdamW/ViT)
- **Early Stopping** : patience=8 epochs
- **Data Augmentation** : RandomHorizontalFlip, RandomRotation(10°), ColorJitter

---

## 👤 Équipe

Projet réalisé dans le cadre du cours de Deep Learning & Machine Learning.
Equipe :
Matisse MARCHAND
Zinedine MEFTAH
Yuba RAHALI
---

## 📄 Références

- MedMNIST v2 : Yang et al., "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification", 2023
- NIH Chest X-rays : Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks", CVPR 2017
- Vision Transformer : Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021
- ResNet : He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
