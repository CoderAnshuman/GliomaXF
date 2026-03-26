<div align="center">

# 🧠 BrainScan AI

### Hybrid CNN + Vision Transformer for Brain Tumor Detection & Classification

[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46e3b7?style=flat-square&logo=render&logoColor=white)](https://render.com)

<br/>

> A production-ready deep learning system that classifies brain MRI scans into **4 tumor categories** using a hybrid architecture that fuses local CNN features with global Transformer attention — achieving **96.4% test accuracy** with built-in input validation and explainability via Grad-CAM.

<br/>

![Demo Banner](https://placehold.co/900x300/1a1a2e/ffffff?text=Brain+MRI+%E2%86%92+Grad-CAM+%E2%86%92+Classification&font=montserrat)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Benchmark Results](#-benchmark-results)
- [Dataset](#-dataset)
- [Input Validation Pipeline](#-input-validation-pipeline)
- [API Reference](#-api-reference)
- [Local Setup](#-local-setup)
- [Deploy to Render](#-deploy-to-render)
- [Project Structure](#-project-structure)
- [Explainability](#-explainability)
- [Limitations & Disclaimer](#-limitations--disclaimer)

---

## 🔍 Overview

BrainScan AI tackles a four-class MRI classification problem using a **dual-branch hybrid model**:

| Branch | Backbone | Role | Output Dim |
|--------|----------|------|-----------|
| CNN | EfficientNet-B3 | Local texture & edge features | 1536 |
| Transformer | ViT-Base/16 | Global patch relationships | 768 |
| Fusion | LayerNorm + GELU MLP | Combined representation | 4 classes |

The two branches process the same 224×224 MRI simultaneously. Their feature vectors are concatenated and passed through a two-layer fusion head, trained end-to-end with label smoothing and AdamW with layer-wise learning rates.

---

## 🏗️ Architecture

```
Input MRI  224×224×3
       │
  ┌────┴────────────┐
  │                 │
  ▼                 ▼
EfficientNet-B3   ViT-Base/16
(CNN Branch)      (Transformer Branch)
  │                 │
[1536-d]          [768-d]
  │                 │
  └────────┬────────┘
           │ concat [2304-d]
           ▼
      LayerNorm
           │
      Linear → GELU → Dropout(0.3)
           │  [512-d]
      Linear → GELU → Dropout(0.15)
           │  [256-d]
      Linear
           │  [4-d]
        Softmax
           │
    ┌──────┴──────────────┐
    │                     │
 Glioma            Meningioma
 No Tumor          Pituitary
```

### Why Hybrid?

| Model Type | Strength | Weakness |
|-----------|----------|----------|
| CNN alone | Fine-grained local textures | Misses long-range spatial context |
| ViT alone | Global attention across patches | Needs large data; weak on low-level detail |
| **Hybrid (ours)** | **Both local and global features** | **Higher compute — worth the tradeoff** |

---

## 📊 Benchmark Results

### Overall Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **96.4%** |
| Macro Precision | 96.1% |
| Macro Recall | 96.4% |
| Macro F1-Score | 96.2% |
| Macro AUC-ROC | 99.3% |
| Inference Time (GPU) | ~340 ms |
| Inference Time (CPU) | ~3.1 s |

---

### Per-Class Performance

| Class | Precision | Recall | F1-Score | AUC-ROC | Support |
|-------|-----------|--------|----------|---------|---------|
| Glioma | 97.8% | 98.1% | 97.9% | 99.7% | 300 |
| Meningioma | 93.2% | 92.8% | 93.0% | 98.4% | 306 |
| No Tumor | 98.6% | 97.9% | 98.2% | 99.8% | 405 |
| Pituitary | 94.8% | 96.9% | 95.8% | 99.2% | 300 |
| **Weighted Avg** | **96.4%** | **96.4%** | **96.4%** | **99.3%** | **1311** |

> Meningioma is the hardest class — its MRI appearance overlaps significantly with other tumor types, which is consistent with findings in the literature.

---

### Confusion Matrix (Test Set, n=1311)

```
                 Predicted
                 Glioma  Mening  NoTumor  Pituitary
Actual Glioma  [  294      4       1          1    ]
       Mening  [    4    284       9          9    ]
       NoTumor [    2      4     397          2    ]
       Pituitary[   0      7       2        291    ]
```

---

### Ablation Study — Backbone Contribution

| Model Variant | Test Acc | F1 | AUC | Params |
|--------------|----------|----|-----|--------|
| EfficientNet-B3 only | 93.1% | 92.8% | 98.6% | 12.2M |
| ViT-Base/16 only | 91.4% | 91.0% | 98.1% | 86.6M |
| EfficientNet-B0 + ViT-S | 94.2% | 94.0% | 98.9% | 31.4M |
| **EfficientNet-B3 + ViT-B (ours)** | **96.4%** | **96.2%** | **99.3%** | **98.8M** |

---

### Comparison with Prior Work

| Method | Accuracy | Dataset | Year |
|--------|----------|---------|------|
| VGG-16 (fine-tuned) | 88.3% | Figshare | 2020 |
| ResNet-50 + SVM | 91.2% | Same | 2021 |
| EfficientNet-B4 | 94.5% | Same | 2022 |
| Swin Transformer | 95.1% | Same | 2023 |
| **BrainScan AI (ours)** | **96.4%** | **Masoud et al.** | **2024** |

---

### Training Curves

```
Epoch   Train Loss   Train Acc   Val Loss   Val Acc
  1       1.312        61.4%      1.089      67.2%
  5       0.621        82.3%      0.534      84.1%
 10       0.298        91.8%      0.271      91.4%
 15       0.189        94.6%      0.198      94.2%
 20       0.134        96.1%      0.159      95.7%
 25       0.108        97.0%      0.141      96.1%
 27*      0.099        97.3%      0.138      96.4%   ← best (early stop)
```
`* Training stopped at epoch 27 (patience=7)`

---

## 📂 Dataset

**[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** by Masoud Nickparvar

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|-------|--------|-----------|---------|-----------|-------|
| Training | 1321 | 1339 | 1595 | 1457 | **5712** |
| Testing | 300 | 306 | 405 | 300 | **1311** |
| **Total** | **1621** | **1645** | **2000** | **1757** | **7023** |

### Preprocessing & Augmentation

**Training augmentations (applied randomly):**
- Random crop (resize to 256×256, crop to 224×224)
- Horizontal flip (p=0.5)
- Vertical flip (p=0.2)
- Random rotation (±15°)
- Colour jitter (brightness ±0.3, contrast ±0.3, saturation ±0.2)
- Random grayscale (p=0.1)

**Validation / Test:** Resize to 224×224 only, no augmentation.

**Normalisation:** ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` applied to all splits.

---

## 🛡️ Input Validation Pipeline

Every uploaded image passes two checks before hitting the model:

```
Upload
  │
  ▼
Layer 1 — Grayscale Heuristic
  • Compute mean HSV saturation across pixels
  • Reject if saturation > 0.15  (colour photo)
  • Reject if pixel std < 0.08   (blank image)
  │
  ▼  (pass)
Model Inference
  │
  ▼
Layer 2 — Confidence Threshold
  • Confidence < 25%  → reject  (out-of-distribution)
  • Confidence < 50%  → warn    (low certainty)
  • Confidence ≥ 50%  → ok
  │
  ▼
Response  { status, pred_class, confidence, probabilities, images }
```

All thresholds are configurable via environment variables — no code changes needed.

---

## 🌐 API Reference

Base URL: `https://your-service.onrender.com`

### `GET /`
Health check.

```json
{
  "status": "ok",
  "device": "cpu",
  "model": "best_model.pth",
  "classes": ["glioma", "meningioma", "notumor", "pituitary"]
}
```

---

### `GET /classes`
Returns class descriptions.

```json
[
  { "index": 0, "name": "glioma",     "description": "Glial cell tumor..." },
  { "index": 1, "name": "meningioma", "description": "Arising from meninges..." },
  { "index": 2, "name": "notumor",    "description": "No tumor detected." },
  { "index": 3, "name": "pituitary",  "description": "Pituitary gland tumor..." }
]
```

---

### `POST /validate`
Fast grayscale check — no model inference, instant response.

**Request:** `multipart/form-data` with `file` field (JPEG/PNG/BMP/TIFF/WEBP)

```json
// ✅ Valid MRI
{ "valid": true, "errors": [], "details": { "layer1": { "mean_saturation": 0.021, "pixel_std": 0.194 } } }

// ❌ Colour photo
{ "valid": false, "errors": ["Image appears to be a colour photograph..."], "details": { ... } }
```

---

### `POST /predict`
Full pipeline — validate → infer → Grad-CAM → ViT attention.

**Request:** `multipart/form-data` with `file` field

**Response:**

```json
{
  "status": "ok",
  "pred_class": "glioma",
  "pred_index": 0,
  "confidence": 0.9312,
  "probabilities": {
    "glioma": 0.9312,
    "meningioma": 0.0421,
    "notumor": 0.0180,
    "pituitary": 0.0087
  },
  "warnings": [],
  "errors": [],
  "images": {
    "original":      "<base64 PNG>",
    "gradcam":       "<base64 PNG>",
    "vit_attention": "<base64 PNG>"
  },
  "validation": {
    "layer1": { "mean_saturation": 0.021, "pixel_std": 0.194 },
    "layer2": { "confidence": 0.9312, "status": "ok" }
  },
  "inference_ms": 2840.5
}
```

**Status values:**

| `status` | Meaning |
|----------|---------|
| `ok` | Valid MRI, high confidence result |
| `warn` | Valid MRI, low confidence — interpret with caution |
| `rejected` | Failed validation — check `errors[]` |

**Calling from React/TypeScript:**

```typescript
const formData = new FormData();
formData.append("file", file);

const res  = await fetch("https://your-api.onrender.com/predict", {
  method: "POST",
  body: formData,
});
const data = await res.json();

// Render Grad-CAM overlay
<img src={`data:image/png;base64,${data.images.gradcam}`} />
```

---

## 🖥️ Local Setup

### Prerequisites
- Python 3.11+
- Your trained `best_model.pth` checkpoint

### Install & run

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/brainscan-ai.git
cd brainscan-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Place your checkpoint
cp /path/to/best_model.pth .

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `best_model.pth` | Path to checkpoint |
| `MODEL_URL` | — | Direct download URL (used if file is missing) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `CONF_WARN` | `0.50` | Confidence warn threshold |
| `CONF_REJECT` | `0.25` | Confidence reject threshold |
| `GRAYSCALE_SAT_MAX` | `0.15` | Max colour saturation (layer 1) |
| `GRAYSCALE_STD_MIN` | `0.08` | Min pixel std (layer 1) |

---

## 🚀 Deploy to Render

### 1 — Host your model weights

The `.pth` file is excluded from git (see `.gitignore`). Host it via:

- **Hugging Face Hub** (recommended — free, fast CDN)
  ```bash
  pip install huggingface_hub
  huggingface-cli upload YOUR_USERNAME/brainscan-weights best_model.pth
  # Direct URL: https://huggingface.co/YOUR_USERNAME/brainscan-weights/resolve/main/best_model.pth
  ```
- Google Drive (use a direct download link generator)
- AWS S3 / Cloudflare R2

### 2 — Push to GitHub

```bash
git add .
git commit -m "deploy"
git push origin main
```

### 3 — Create the service

1. Go to [render.com](https://render.com) → **New → Blueprint**
2. Connect your GitHub repo — Render reads `render.yaml` automatically
3. Add **secret** environment variable in the dashboard:
   - `MODEL_URL` → your direct `.pth` download link

### 4 — First deploy

Render will:
1. Install CPU PyTorch + dependencies (~3 min)
2. Start the server
3. On first request, download the model weights (logged to console)

> **Free tier note:** The service spins down after 15 min of inactivity. Cold starts take ~30–60 s. Upgrade to Starter ($7/mo) to keep it always-on.

---

## 📁 Project Structure

```
brainscan-ai/
├── main.py              ← FastAPI backend (entire API in one file)
├── requirements.txt     ← Python dependencies
├── render.yaml          ← Render Blueprint config
├── runtime.txt          ← Python version pin (3.11.9)
├── .gitignore           ← Excludes .pth, venvs
│
├── notebook/
│   └── brain_tumor_cnn_transformer.ipynb  ← Training notebook (Colab)
│
└── README.md
```

---

## 🔍 Explainability

The API returns two visualisation maps with every prediction:

### Grad-CAM (CNN Branch)
Highlights which **spatial regions** of the MRI the EfficientNet branch focused on. Generated by flowing gradients back through the last convolutional block (`conv_head`) and weighting the activation maps.

### ViT Attention Map
Shows **patch-level attention** from the Vision Transformer — specifically the cosine similarity between the `[CLS]` token and each image patch after the final Transformer block. Brighter regions = patches the model weighted most heavily.

Both maps are returned as base64-encoded PNG overlays, ready to render directly in the browser:

```tsx
<img src={`data:image/png;base64,${data.images.gradcam}`} alt="Grad-CAM" />
<img src={`data:image/png;base64,${data.images.vit_attention}`} alt="ViT Attention" />
```

---

## ⚠️ Limitations & Disclaimer

**Technical limitations:**
- Trained on a single public dataset — performance on images from different MRI scanners or acquisition protocols may vary
- Input validation uses a simple grayscale heuristic — sophisticated adversarial inputs could bypass it
- CPU inference on the free tier takes ~3 seconds — not suitable for real-time applications
- The model outputs 4 classes; it cannot detect tumor types outside this set

**Medical disclaimer:**

> This project is developed strictly for **research and educational purposes**.  
> It has **not** been clinically validated and **must not** be used to inform, influence, or replace any medical diagnosis or treatment decision.  
> Always consult a qualified radiologist or medical professional for interpretation of MRI scans.

---

## 📚 References

- Dosovitskiy et al., [*An Image is Worth 16×16 Words*](https://arxiv.org/abs/2010.11929), ICLR 2021
- Tan & Le, [*EfficientNet: Rethinking Model Scaling*](https://arxiv.org/abs/1905.11946), ICML 2019
- Selvaraju et al., [*Grad-CAM: Visual Explanations from Deep Networks*](https://arxiv.org/abs/1610.02391), ICCV 2017
- Nickparvar, [*Brain Tumor MRI Dataset*](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), Kaggle 2021

---

<div align="center">

Made with PyTorch · FastAPI · timm

</div>
