"""
╔══════════════════════════════════════════════════════════════════╗
║         Brain Tumor Detection — Backend Server                   ║
║         Hybrid CNN (EfficientNet-B3) + ViT Model                 ║
║                                                                  ║
║  HOW TO RUN:                                                     ║
║    1. Place your best_model.pth in the same folder as this file  ║
║    2. python server.py                                           ║
║    3. API runs at http://localhost:8000                          ║
║    4. API docs at  http://localhost:8000/docs                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0 — Auto-install missing packages (runs only when needed)
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, sys

def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *pkgs])

try:
    import torch
except ImportError:
    print("[setup] Installing PyTorch...")
    _pip("torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121")

try:
    import timm
except ImportError:
    print("[setup] Installing timm...")
    _pip("timm")

try:
    import clip
except ImportError:
    print("[setup] Installing CLIP...")
    _pip("git+https://github.com/openai/CLIP.git")

try:
    import fastapi
except ImportError:
    print("[setup] Installing FastAPI + uvicorn...")
    _pip("fastapi", "uvicorn[standard]", "python-multipart")

try:
    import google.genai as genai
except ImportError:
    print("[setup] Installing Google GenAI...")

# remaining stdlib-adjacent deps
try:
    import PIL
except ImportError:
    _pip("Pillow")

try:
    import matplotlib
except ImportError:
    _pip("matplotlib")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Imports (all guaranteed installed now)
# ─────────────────────────────────────────────────────────────────────────────
import io, base64, colorsys, logging, time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import clip
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")                   # headless — no display needed
from matplotlib import colormaps       # only used for jet/inferno LUTs — no Figure/Axes overhead
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai
import os

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  ←  Edit this block to customise behaviour
# ─────────────────────────────────────────────────────────────────────────────

from huggingface_hub import hf_hub_download

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

MODEL_PATH = hf_hub_download(
    repo_id="AnshumanShukla/Gliomax",
    filename="best_model.pth"
)   
    

PORT = int(os.environ.get("PORT", 7860))

# If True, requests from any origin are accepted (fine for local dev).
# Set to your React dev URL (e.g. "http://localhost:5173") for tighter security.
CORS_ORIGINS = ["*"]

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = 224
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)
MEAN        = [0.485, 0.456, 0.406]    # ImageNet stats — must match training
STD         = [0.229, 0.224, 0.225]

# Suggested next steps based on prediction
SUGGESTED_STEPS = {
    "glioma": [
        "Consult with neuro-oncologist immediately",
        "Schedule MRI with contrast for detailed assessment",
        "Consider biopsy if not already done",
        "Discuss treatment options (surgery, radiation, chemotherapy)"
    ],
    "meningioma": [
        "Consult with neurosurgeon",
        "Schedule follow-up imaging in 3-6 months",
        "Monitor for symptom progression",
        "Consider surgical resection if symptomatic"
    ],
    "notumor": [
        "No immediate action required",
        "Regular health check-ups",
        "Report any new symptoms to healthcare provider",
        "Continue routine screening as recommended"
    ],
    "pituitary": [
        "Consult with endocrinologist",
        "Hormone level testing",
        "Schedule pituitary-specific MRI",
        "Monitor for hormonal imbalances"
    ]
}

# ── Validation thresholds ─────────────────────────────────────────────────────
#   Raise CLIP_MRI_THRESHOLD if too many non-MRI images slip through.
#   Lower GRAYSCALE_SAT_THRESHOLD if real MRIs are being rejected.

GRAYSCALE_SAT_THRESHOLD     = 0.15     # Layer 1: max mean colour saturation
GRAYSCALE_STD_THRESHOLD     = 0.08     # Layer 1: min pixel intensity std (rejects blank images)
CLIP_MRI_THRESHOLD          = 0.60     # Layer 2: min CLIP MRI score fraction (0.5 = neutral, see _layer2_clip)
CONFIDENCE_WARN_THRESHOLD   = 0.50     # Layer 3: confidence below this → warning
CONFIDENCE_REJECT_THRESHOLD = 0.25     # Layer 3: confidence below this → reject

# ── CLIP prompts ──────────────────────────────────────────────────────────────
CLIP_POSITIVE = [
    "a brain MRI scan",
    "a brain CT scan",
    "a magnetic resonance image of the human brain",
    "a medical scan showing brain tissue",
    "a grayscale MRI of a human head",
    "a neurological scan showing brain structure",
]
CLIP_NEGATIVE = [
    "a photo of a random object",
    "a natural photograph",
    "a photo of an animal",
    "a photo of food",
    "a photo of a landscape",
    "a photo of a person",
    "a screenshot or chart",
    "a photo of a vehicle",
    "a black and white photo of a person"
]

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("brain_tumor_api")

# ─────────────────────────────────────────────────────────────────────────────
#  Model Architecture  (must exactly match the notebook training code)
# ─────────────────────────────────────────────────────────────────────────────

class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=False, num_classes=0, global_pool="avg"
        )
        self.out_dim = self.backbone.num_features           # 1536

    def forward(self, x):
        return self.backbone(x)


class ViTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0
        )
        self.out_dim = self.backbone.embed_dim              # 768

    def forward(self, x):
        return self.backbone(x)


class FusionHead(nn.Module):
    def __init__(self, cnn_dim, vit_dim, fusion_dim=512, num_classes=4, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cnn_dim + vit_dim),
            nn.Linear(cnn_dim + vit_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(self, cnn_feat, vit_feat):
        return self.net(torch.cat([cnn_feat, vit_feat], dim=-1))


class HybridCNNViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = CNNBranch()
        self.vit  = ViTBranch()
        self.head = FusionHead(self.cnn.out_dim, self.vit.out_dim)

    def forward(self, x):
        return self.head(self.cnn(x), self.vit(x))


# ─────────────────────────────────────────────────────────────────────────────
#  Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Runs the hybrid model exactly ONCE per request (one forward + one backward)
    and captures everything downstream needs from that single pass:
      - CNN activations/gradients -> Grad-CAM
      - final ViT block's tokens  -> attention map
      - logits                   -> class probabilities
    Previously these were computed via 3 separate forward passes (one of them
    a hand-rolled re-implementation of the ViT forward). That tripling was the
    dominant cost in the 3-4s latency. It also meant a silent failure in any
    one of the three passes (e.g. gradients never populating) produced a
    blank/wrong image with no error surfaced.
    """
    def __init__(self, model: HybridCNNViT):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self.vit_tokens  = None

        cnn_tgt = model.cnn.backbone.conv_head       # last conv block of EfficientNet-B3
        cnn_tgt.register_forward_hook(self._save_cnn_activation)
        cnn_tgt.register_full_backward_hook(self._save_cnn_gradient)

        vit_tgt = model.vit.backbone.blocks[-1]      # output of final transformer block
        vit_tgt.register_forward_hook(self._save_vit_tokens)

    def _save_cnn_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_cnn_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def _save_vit_tokens(self, module, inp, out):
        self.vit_tokens = out.detach()

    def __call__(self, img_t: torch.Tensor):
        """
        img_t: (1, 3, H, W)
        Returns: cam (H,W numpy 0-1), vit_map (H,W numpy 0-1), probs (numpy), pred_class_idx (int)
        """
        self.model.eval()
        img_t = img_t.to(DEVICE)
        img_t.requires_grad_(True)

        logits    = self.model(img_t)
        class_idx = int(logits.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            # Fails loudly now instead of silently rendering a blank overlay.
            raise RuntimeError(
                "Grad-CAM hooks did not fire — check that model.cnn.backbone.conv_head "
                "is on the path from input to logits."
            )

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # ViT attention proxy — reuses tokens captured during the SAME forward
        # pass above, instead of re-running patch_embed + blocks a second time.
        tokens  = self.vit_tokens                              # (1, 1+P, D)
        sim     = F.cosine_similarity(tokens[:, 0:1], tokens[:, 1:], dim=-1)
        sim     = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        n_p     = int(sim.shape[-1] ** 0.5)
        vit_map = F.interpolate(
            sim.reshape(1, 1, n_p, n_p).float(),
            size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()

        probs = F.softmax(logits.detach(), dim=1).squeeze().cpu().numpy()

        return cam, vit_map, probs, class_idx


# ─────────────────────────────────────────────────────────────────────────────
#  Global model state  (populated once at startup, reused for every request)
# ─────────────────────────────────────────────────────────────────────────────

_S: dict = {}                           # _S["model"], _S["grad_cam"], etc.

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def _startup():
    """Load tumor model + CLIP once at server start."""
    log.info(f"Device: {DEVICE}")

    # ── Tumor model ──────────────────────────────────────────────────────────
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"\n\n  ❌  Checkpoint not found: {MODEL_PATH}\n"
            f"      Copy your best_model.pth next to server.py and restart.\n"
        )
    log.info(f"Loading HybridCNNViT from '{MODEL_PATH}'...")
    model = HybridCNNViT().to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    # Handles both raw state_dict and wrapped {"model_state_dict": ...} format
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    log.info("✅ Tumor model ready")

    # ── CLIP ─────────────────────────────────────────────────────────────────
    log.info("Loading CLIP (ViT-B/32) — first run downloads ~350 MB...")
    clip_model, clip_prep = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    all_prompts = CLIP_POSITIVE + CLIP_NEGATIVE
    with torch.no_grad():
        toks      = clip.tokenize(all_prompts).to(DEVICE)
        text_feat = clip_model.encode_text(toks)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    log.info("✅ CLIP ready")

    _S["model"]      = model
    _S["grad_cam"]   = GradCAM(model)
    _S["clip"]       = clip_model
    _S["clip_prep"]  = clip_prep
    _S["text_feat"]  = text_feat
    _S["n_pos"]      = len(CLIP_POSITIVE)

# ── Gemini ───────────────────────────────────────────────────────────────

if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    log.info("Initializing Gemini AI...")
    # Initialize the new Client instead of using .configure()
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Store the client in your settings dictionary
    _S["gemini"] = client
    log.info("✅ Gemini ready")
else:
    log.warning("Gemini API key not set, chat functionality will be disabled")
    _S["gemini"] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Validation — Layer 1: Grayscale heuristic
# ─────────────────────────────────────────────────────────────────────────────

def _layer1_grayscale(img: Image.Image):
    """
    Returns (passed: bool, message: str, details: dict)
    Brain MRIs are near-grayscale (low HSV saturation).
    Vectorised HSV-saturation calc (S = (max-min)/max) over every pixel —
    replaces a pure-Python colorsys.rgb_to_hsv() loop, which was both slower
    and (at 1-in-4 sampling) noisier than just vectorising over all pixels.
    """
    arr      = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    mx       = arr.max(axis=-1)
    mn       = arr.min(axis=-1)
    sat      = np.where(mx > 0, (mx - mn) / (mx + 1e-8), 0.0)
    mean_sat = float(sat.mean())
    px_std   = float(np.mean(arr, axis=2).std())
    details  = {"mean_saturation": round(mean_sat, 4), "pixel_std": round(px_std, 4)}

    if mean_sat > GRAYSCALE_SAT_THRESHOLD:
        return False, (
            f"Image appears to be a colour photograph, not a grayscale MRI. "
            f"Mean saturation: {mean_sat:.3f} (limit: {GRAYSCALE_SAT_THRESHOLD}). "
            f"Please upload a brain MRI scan."
        ), details

    if px_std < GRAYSCALE_STD_THRESHOLD:
        return False, (
            f"Image appears blank or uniform. "
            f"Pixel std: {px_std:.3f} (minimum: {GRAYSCALE_STD_THRESHOLD}). "
            f"Please upload a real MRI image."
        ), details

    return True, "Layer 1 passed", details


# ─────────────────────────────────────────────────────────────────────────────
#  Validation — Layer 2: CLIP MRI similarity
# ─────────────────────────────────────────────────────────────────────────────

def _layer2_clip(img: Image.Image):
    """
    Returns (passed: bool, message: str, details: dict)
    Compares image to brain MRI vs. random-object text prompts via CLIP.

    Two bugs fixed here:
    1. Missing temperature scaling. Raw cosine similarities from CLIP sit in a
       narrow band (~0.02-0.05 apart between related/unrelated prompts), so a
       plain softmax over them came out nearly flat regardless of the image —
       CLIP's own logit_scale (~100x) is what makes zero-shot classification
       discriminative at all, and it was never applied.
    2. Prompt-count bias. Summing softmax mass over 6 positive vs 9 negative
       prompts means even a *flat* distribution gives ~40% to "MRI" — comfortably
       above the old 0.20 threshold. Averaging similarity per class first (2-way
       softmax) removes the count bias so 0.5 is the true neutral point.
    """
    inp = _S["clip_prep"](img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = _S["clip"].encode_image(inp)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        sims = (feat @ _S["text_feat"].T).squeeze(0).cpu().float().numpy()

    n           = _S["n_pos"]
    logit_scale = float(_S["clip"].logit_scale.exp().item())   # CLIP's own temperature (~100)
    mean_pos    = float(sims[:n].mean())
    mean_neg    = float(sims[n:].mean())

    pair        = np.array([mean_pos, mean_neg]) * logit_scale
    pair        = np.exp(pair - pair.max())
    mri_frac    = float(pair[0] / pair.sum())
    top_neg     = CLIP_NEGATIVE[int(sims[n:].argmax())]
    top_pos     = CLIP_POSITIVE[int(sims[:n].argmax())]
    details     = {
        "mri_softmax_fraction"  : round(mri_frac, 4),
        "mri_similarity_mean"   : round(mean_pos, 4),
        "other_similarity_mean" : round(mean_neg, 4),
        "top_positive_match"    : top_pos,
        "top_negative_match"    : top_neg,
    }

    if mri_frac < CLIP_MRI_THRESHOLD:
        return False, (
            f"Image does not resemble a brain MRI. "
            f"CLIP MRI score: {mri_frac:.3f} (minimum: {CLIP_MRI_THRESHOLD}). "
            f"Most similar to: '{top_neg}'. "
            f"Please upload a valid brain MRI scan."
        ), details

    return True, f"Layer 2 passed (CLIP score: {mri_frac:.3f})", details


# ─────────────────────────────────────────────────────────────────────────────
#  Validation — Layer 3: Model confidence
# ─────────────────────────────────────────────────────────────────────────────

def _layer3_confidence(conf: float):
    """Returns (status: 'ok'|'warn'|'reject', message: str)"""
    if conf < CONFIDENCE_REJECT_THRESHOLD:
        return "reject", (
            f"Model confidence critically low: {conf*100:.1f}% "
            f"(minimum: {CONFIDENCE_REJECT_THRESHOLD*100:.0f}%). "
            "Image may be an unusual or low-quality scan."
        )
    if conf < CONFIDENCE_WARN_THRESHOLD:
        return "warn", (
            f"Model confidence low: {conf*100:.1f}% "
            f"(warn threshold: {CONFIDENCE_WARN_THRESHOLD*100:.0f}%). "
            "Interpret with caution."
        )
    return "ok", f"Confidence OK ({conf*100:.1f}%)"


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────

def _to_png_b64(rgb_uint8: np.ndarray) -> str:
    """Encode an (H,W,3) uint8 array as base64 PNG — no matplotlib Figure/Axes."""
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_png(rgb_hw3: np.ndarray) -> str:
    """Encode a plain (H,W,3) float [0,1] image."""
    return _to_png_b64((np.clip(rgb_hw3, 0, 1) * 255).astype(np.uint8))


def _overlay_b64(base_hw3: np.ndarray, mask_hw: np.ndarray, cmap_name: str, alpha: float) -> str:
    """Alpha-blend a colormapped heatmap over the base image, pure numpy/PIL."""
    cmap = colormaps[cmap_name]
    heat = cmap(np.clip(mask_hw, 0, 1))[:, :, :3]                       # (H,W,3) float 0-1
    blended = np.clip(base_hw3 * (1 - alpha) + heat * alpha, 0, 1)
    return _to_png_b64((blended * 255).astype(np.uint8))


def _run_inference(img: Image.Image) -> dict:
    img_t = _transform(img.convert("RGB")).unsqueeze(0)     # (1, 3, 224, 224)

    # Single forward + single backward — produces Grad-CAM, ViT attention,
    # and class probabilities all from one pass through the model.
    cam, vit_map, probs, pred_idx = _S["grad_cam"](img_t)

    # Denormalise image for visualisation
    img_np = img_t.squeeze().permute(1, 2, 0).detach().numpy()
    img_np = np.clip(img_np * np.array(STD) + np.array(MEAN), 0, 1)

    return {
        "pred_class"    : CLASS_NAMES[pred_idx],
        "pred_index"    : int(pred_idx),
        "confidence"    : float(probs[pred_idx]),
        "probabilities" : {CLASS_NAMES[i]: round(float(probs[i]), 6) for i in range(NUM_CLASSES)},
        "suggestedNextSteps": SUGGESTED_STEPS[CLASS_NAMES[pred_idx]],
        "images": {
            "original"     : _b64_png(img_np),
            "gradcam"      : _overlay_b64(img_np, cam,     "jet",     0.45),
            "vit_attention": _overlay_b64(img_np, vit_map, "inferno", 0.50),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    _startup()
    yield
    _S.clear()

app = FastAPI(
    title       = "Brain Tumor Detection API",
    description = "Hybrid CNN+ViT with 3-layer input validation",
    version     = "1.0.0",
    lifespan    = _lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = CORS_ORIGINS,
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Response schemas ──────────────────────────────────────────────────────────

class ValidationDetail(BaseModel):
    layer1: Optional[dict] = None
    layer2: Optional[dict] = None
    layer3: Optional[dict] = None

class PredictResponse(BaseModel):
    status        : str                     # "ok" | "warn" | "rejected"
    pred_class    : Optional[str]  = None
    pred_index    : Optional[int]  = None
    confidence    : Optional[float]= None
    probabilities : Optional[dict] = None
    suggestedNextSteps: Optional[list[str]] = None
    validation    : ValidationDetail
    warnings      : list[str]
    errors        : list[str]
    images        : Optional[dict] = None   # keys: original, gradcam, vit_attention (base64 PNGs)
    inference_ms  : Optional[float]= None

class ValidateResponse(BaseModel):
    valid    : bool
    errors   : list[str]
    warnings : list[str]
    details  : dict

class ChatRequest(BaseModel):
    message: str
    history: Optional[list[dict]] = None

class ChatResponse(BaseModel):
    response: str
    error: Optional[str] = None
    
# ── Allowed MIME types ────────────────────────────────────────────────────────

_ALLOWED = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}

def _assert_image(f: UploadFile):
    if f.content_type not in _ALLOWED:
        raise HTTPException(415, f"Unsupported type '{f.content_type}'. Allowed: JPEG PNG BMP TIFF WEBP.")

def _open_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Cannot read image: {e}")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/health", summary="Server health check")
def health():
    return {
        "status" : "ok",
        "device" : str(DEVICE),
        "model"  : Path(MODEL_PATH).name,
        "classes": CLASS_NAMES,
    }


@app.get("/classes", summary="List supported tumor classes")
def get_classes():
    desc = {
        "glioma"     : "Glial cell tumor — can be benign or malignant.",
        "meningioma" : "Arising from meninges — usually benign.",
        "notumor"    : "No tumor detected.",
        "pituitary"  : "Pituitary gland tumor — often benign.",
    }
    return [{"index": i, "name": c, "description": desc[c]} for i, c in enumerate(CLASS_NAMES)]


@app.post("/validate-only", response_model=ValidateResponse,
          summary="Fast input check — layers 1 & 2 only, no model inference")
async def validate_only(file: UploadFile = File(...)):
    _assert_image(file)
    img     = _open_image(await file.read())
    errors, warnings, details = [], [], {}

    ok, msg, det = await asyncio.to_thread(_layer1_grayscale, img)
    details["layer1"] = det
    if not ok:
        return ValidateResponse(valid=False, errors=[msg], warnings=[], details=details)

    ok, msg, det = await asyncio.to_thread(_layer2_clip, img)
    details["layer2"] = det
    if not ok:
        return ValidateResponse(valid=False, errors=[msg], warnings=[], details=details)

    return ValidateResponse(valid=True, errors=[], warnings=[], details=details)


@app.post("/predict", response_model=PredictResponse,
          summary="Full pipeline — validate (3 layers) + inference + Grad-CAM")
async def predict(file: UploadFile = File(...)):
    _assert_image(file)
    img             = _open_image(await file.read())
    errors, warnings, vdet = [], [], {}

    # Layer 1
    ok, msg, det = await asyncio.to_thread(_layer1_grayscale, img)
    vdet["layer1"] = det
    if not ok:
        return PredictResponse(status="rejected", validation=ValidationDetail(**vdet),
                               warnings=[], errors=[msg], suggestedNextSteps=None)

    # Layer 2
    ok, msg, det = await asyncio.to_thread(_layer2_clip, img)
    vdet["layer2"] = det
    if not ok:
        return PredictResponse(status="rejected", validation=ValidationDetail(**vdet),
                               warnings=[], errors=[msg], suggestedNextSteps=None)

    # Inference — single forward+backward pass (see GradCAM class docstring)
    t0  = time.perf_counter()
    res = await asyncio.to_thread(_run_inference, img)
    ms  = round((time.perf_counter() - t0) * 1000, 1)

    # Layer 3
    conf_status, conf_msg = _layer3_confidence(res["confidence"])
    vdet["layer3"] = {"confidence": res["confidence"], "status": conf_status, "message": conf_msg}

    if conf_status == "reject":
        return PredictResponse(
            status="rejected",
            pred_class=res["pred_class"], pred_index=res["pred_index"],
            confidence=res["confidence"], probabilities=res["probabilities"],
            suggestedNextSteps=None,
            validation=ValidationDetail(**vdet),
            warnings=[], errors=[conf_msg], inference_ms=ms,
        )

    if conf_status == "warn":
        warnings.append(conf_msg)

    return PredictResponse(
        status        = "warn" if warnings else "ok",
        pred_class    = res["pred_class"],
        pred_index    = res["pred_index"],
        confidence    = res["confidence"],
        probabilities = res["probabilities"],
        suggestedNextSteps = res["suggestedNextSteps"],
        validation    = ValidationDetail(**vdet),
        warnings      = warnings,
        errors        = [],
        images        = res["images"],
        inference_ms  = ms,
    )

@app.post("/chat", response_model=ChatResponse,
          summary="Chat with Gemini AI about brain tumor diagnostics")
async def chat(request: ChatRequest):
    if not _S.get("gemini"):
        return ChatResponse(response="", error="Gemini AI not configured. Please set GEMINI_API_KEY environment variable.")

    try:
        model = _S["gemini"]
        history = request.history or []
        
        # Convert history format if needed
        chat_history = []
        for msg in history:
            if msg.get("role") == "user":
                chat_history.append({"role": "user", "parts": [{"text": msg.get("content", "")}]})
            elif msg.get("role") == "model":
                chat_history.append({"role": "model", "parts": [{"text": msg.get("content", "")}]})

        chat = model.start_chat(history=chat_history)
        
        system_instruction = """You are Gliomax AI, a medical assistant for neuro-oncologists. You provide information about brain tumor diagnostics (Glioma, Meningioma, Pituitary, and No Tumor), our CNN-Transformer architecture, and clinical performance. Be precise, clinical, and helpful. Use markdown for formatting. If asked about your creators, mention Anshuman Shukla as the Leader from Parul University."""
        
        response = chat.send_message(f"{system_instruction}\n\nUser: {request.message}")
        return ChatResponse(response=response.text)
    except Exception as e:
        log.error(f"Chat error: {e}")
        return ChatResponse(response="", error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    HOST = "0.0.0.0"
    PORT = int(os.environ.get("PORT", 7860))

    print("\n" + "="*60)
    print("  🧠  Brain Tumor Detection API")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Device  : {DEVICE}")
    print(f"  URL     : http://localhost:{PORT}")
    print(f"  API docs: http://localhost:{PORT}/docs")
    print("="*60 + "\n")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")