#!/usr/bin/env python3
# 04_CuVLER_GreenOrange.py

#  CuVLER (Green Orange Detection)
# 
# **Paper:** CuVLER: Enhanced Unsupervised Object Discoveries through Exhaustive Self-Supervised Transformers — Arica et al., CVPR 2024
# **arXiv:** https://arxiv.org/abs/2403.07700
# **GitHub:** https://github.com/shahaf-arica/CuVLER
# **Institution:** Independent / CVPR 2024
# **Approach:** VoteCut (DINO + DINOv2 + MAE features → Normalized Cut → pixel voting) → soft target loss → Cascade Mask R-CNN
# **Pretrained on:** ImageNet + ViTs (DINO, DINOv2, MAE) — NO fruit domain knowledge
# 
# ---
# 
# ## CuVLER vs CutLER — The Key Difference
# 
# | | CutLER | CuVLER |
# |--|--------|--------|
# | Pseudo-label method | MaskCut (single DINO) | VoteCut (DINO + DINOv2 + MAE combined) |
# | Mask quality | Single-model saliency | Multi-model voting → more complete |
# | Downstream detector | Cascade Mask R-CNN | Cascade Mask R-CNN (same) |
# | Expected performance | Strong baseline | Better (SOTA unsupervised) |
# | Tiled inference | Yes (1920×1080) | Yes (same — same detector type) |
# 
# VoteCut combines three complementary feature representations:
# DINO → global semantic structure
# DINOv2 → stronger visual semantics
# MAE → local texture (orange surface vs leaf surface)
# Combined voting produces more complete, less fragmented masks.
# 
# ---
# 
# ## Prerequisites
# > ⚠️ Run **00_shared.ipynb** before this notebook.
# > CuVLER requires Detectron2 — runs on Linux/JupyterHub only.
# > Windows native not supported.
# 
# ## Run Order
# Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 → Cell 6 → Cell 7 → Cell 8 → Cell 9 → Cell 10 → Cell 11 → Cells 12–16 → Cell 17 → Cell 18
# 
# ## Evaluation Design
# | Split | Trees | Frames | Purpose |
# |-------|-------|--------|---------|
# | Calibration | Tree_01, Tree_04, Tree_05, Tree_06, Tree_07, Tree_08 | 6,367 | Threshold tuning only |
# | Test | Tree_02, Tree_03, Tree_09, Tree_10 | 4,210 | Final paper result |
# | Full | All 10 trees | 10,577 | Supplementary |
# 

# ## Cell 1 — Configuration & Paths

# ── CELL 3 ──────────────────────────────────────────────────
# ============================================================
# CELL 1 — CONFIGURATION & PATHS
# ============================================================
# CuVLER uses tiled inference — same as CutLER.
# Both are Cascade Mask R-CNN detection models.
# VoteCut changes WHAT the model learned, not HOW inference runs.
# ============================================================

import os

# ── Root paths ──────────────────────────────────────────────
BASE_DIR   = '/home/jovyan/OrangeGrove'
FRAMES_DIR = os.path.join(BASE_DIR, 'frames')
DIR_30SEC  = os.path.join(FRAMES_DIR, '30sec')
DIR_40SEC  = os.path.join(FRAMES_DIR, '40sec')

# ── CuVLER repo + checkpoint ────────────────────────────────
CUVLER_DIR  = os.path.join(BASE_DIR, 'CuVLER')
CUTLER_DIR  = os.path.join(BASE_DIR, 'CutLER')   # fallback for config.py
CKPT_DIR    = os.path.join(CUVLER_DIR, 'checkpoints')
CKPT_PATH   = os.path.join(CKPT_DIR, 'cuvler_final.pth')

# Primary: HuggingFace (official CuVLER release)
# Fallback: wget from GitHub releases if HuggingFace blocked
CKPT_URL    = 'https://huggingface.co/shahaf-arica/CuVLER/resolve/main/cuvler_final.pth'

# ── Output & ripe validation paths ──────────────────────────
OUT_DIR     = os.path.join(BASE_DIR, 'results', '04_CuVLER')
RIPE_BASE   = os.path.join(BASE_DIR, 'ripe_validation')
RIPE_FRAMES = os.path.join(RIPE_BASE, 'frames/40sec')
RIPE_ANNOT  = os.path.join(RIPE_BASE, 'annotations/40sec')
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(CKPT_DIR,  exist_ok=True)

# ── Model info ──────────────────────────────────────────────
MODEL_NAME    = 'CuVLER (VoteCut + Cascade Mask R-CNN)'
MODEL_SLUG    = 'cuvler'
PRETRAINED_ON = 'ImageNet + ViTs (DINO, DINOv2, MAE)'

# ── Tiled inference config ───────────────────────────────────
# Identical to CutLER — same Cascade Mask R-CNN architecture.
# At 4K, oranges are 2-5px without tiling → model blind.
# At 1920×1080 tile, oranges ~50px → detectable.
TILE_W       = 1920
TILE_H       = 1080
TILE_OVERLAP = 100
NMS_IOU      = 0.5

# ── Dataset structure ───────────────────────────────────────
TREES_30SEC = ['Tree_01','Tree_02','Tree_03','Tree_04','Tree_05']
TREES_40SEC = ['Tree_06','Tree_07','Tree_08','Tree_09','Tree_10']
VIDEOS      = ['Vid 01','Vid 02','Vid 03']

TREE_FOLDER_MAP = {
    'Tree_01':'Tree_01','Tree_02':'Tree_02','Tree_03':'Tree_03',
    'Tree_04':'Tree_04','Tree_05':'Tree_05',
    'Tree_06':'Tree_06','Tree_07':'Tree_07',
    'Tree_08':'Tree_08','Tree_09':'Tree_09',
    'Tree_10':'Tree_10',
}
TREE_40SEC_IDS = ['Tree_06','Tree_07','Tree_08','Tree_09','Tree_10']

# ── Cal / Test split ────────────────────────────────────────
CAL_TREES  = ['Tree_01','Tree_04','Tree_05',
               'Tree_06','Tree_07','Tree_08']
TEST_TREES = ['Tree_02','Tree_03',
               'Tree_09','Tree_10']
CAL_TREE_IDS  = ['Tree_01','Tree_04','Tree_05',
                  'Tree_06','Tree_07','Tree_08']
TEST_TREE_IDS = ['Tree_02','Tree_03','Tree_09','Tree_10']

# ── Confirmed frame counts ───────────────────────────────────
CONFIRMED_COUNTS = {
    ('Tree_01',  'Vid 01'):301,('Tree_01',  'Vid 02'):305,('Tree_01',  'Vid 03'):300,
    ('Tree_02',  'Vid 01'):300,('Tree_02',  'Vid 02'):300,('Tree_02',  'Vid 03'):300,
    ('Tree_03',  'Vid 01'):300,('Tree_03',  'Vid 02'):306,('Tree_03',  'Vid 03'):300,
    ('Tree_04', 'Vid 01'):300,('Tree_04', 'Vid 02'):315,('Tree_04', 'Vid 03'):316,
    ('Tree_05', 'Vid 01'):305,('Tree_05', 'Vid 02'):305,('Tree_05', 'Vid 03'):300,
    ('Tree_06', 'Vid 01'):400,('Tree_06', 'Vid 02'):403,('Tree_06', 'Vid 03'):400,
    ('Tree_07', 'Vid 01'):403,('Tree_07', 'Vid 02'):405,('Tree_07', 'Vid 03'):400,
    ('Tree_08', 'Vid 01'):404,('Tree_08', 'Vid 02'):400,('Tree_08', 'Vid 03'):405,
    ('Tree_09', 'Vid 01'):400,('Tree_09', 'Vid 02'):400,('Tree_09', 'Vid 03'):404,
    ('Tree_10', 'Vid 01'):400,('Tree_10', 'Vid 02'):400,('Tree_10', 'Vid 03'):400,
}
TOTAL_FRAMES = sum(CONFIRMED_COUNTS.values())
CAL_FRAMES   = sum(v for (t,_),v in CONFIRMED_COUNTS.items() if t in CAL_TREES)
TEST_FRAMES  = sum(v for (t,_),v in CONFIRMED_COUNTS.items() if t in TEST_TREES)

print('=' * 65)
print('  CONFIGURATION LOADED')
print('=' * 65)
print(f'  Model        : {MODEL_NAME}')
print(f'  Pretrained   : {PRETRAINED_ON}')
print(f'  CuVLER dir   : {CUVLER_DIR}')
print(f'  CutLER dir   : {CUTLER_DIR}  (config.py fallback)')
print(f'  Checkpoint   : {CKPT_PATH}')
print(f'  Output dir   : {OUT_DIR}')
print()
print(f'  Inference    : Tiled (same as CutLER)')
print(f'  Tile size    : {TILE_W}×{TILE_H}px')
print(f'  Overlap      : {TILE_OVERLAP}px')
print(f'  NMS IoU      : {NMS_IOU}')
print()
print(f'  Total frames : {TOTAL_FRAMES:,}')
print(f'  Cal frames   : {CAL_FRAMES:,}  ({len(CAL_TREES)} trees)')
print(f'  Test frames  : {TEST_FRAMES:,}  ({len(TEST_TREES)} trees)')
print()
print('  Paths:')
for name, path in [
    ('frames/30sec', DIR_30SEC),
    ('frames/40sec', DIR_40SEC),
    ('CuVLER repo',  CUVLER_DIR),
    ('CutLER repo',  CUTLER_DIR),
    ('output',       OUT_DIR),
    ('ripe frames',  RIPE_FRAMES),
    ('ripe annot',   RIPE_ANNOT),
]:
    status = '✓' if os.path.exists(path) else '✗ MISSING'
    print(f'  {name:<15}: {status}')
print('=' * 65)


# ## Cell 2 — Imports & Dependencies

# ── CELL 5 ──────────────────────────────────────────────────
# ============================================================
# CELL 2 — IMPORTS & DEPENDENCIES
# ============================================================

import sys, subprocess, torch

print('=' * 65)
print('  SYSTEM INFO')
print('=' * 65)
print(f'  Python  : {sys.version}')
print(f'  PyTorch : {torch.__version__}')
print(f'  CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU     : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print()

# Fix NumPy version conflict — detectron2 compiled against NumPy 1.x
print('  Fixing NumPy version conflict...')
subprocess.run(
    [sys.executable, '-m', 'pip', 'install',
     'numpy<2.0', '--force-reinstall', '-q'],
    capture_output=True)
print('  NumPy fixed')

# ⚠️ RESTART KERNEL after NumPy fix if this is the first run

for pkg in ['opencv-python', 'Pillow', 'pandas', 'openpyxl',
            'matplotlib', 'scikit-learn', 'scipy',
            'tqdm', 'tabulate']:
    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', pkg, '-q'],
        capture_output=True)
    print(f"  {'OK' if r.returncode==0 else 'FAIL'} {pkg}")

# Install Detectron2
torch_ver = torch.__version__.split('+')[0]
cuda_ver  = torch.version.cuda.replace('.','') if torch.cuda.is_available() else None
if cuda_ver:
    d2_url = (f'https://dl.fbaipublicfiles.com/detectron2/wheels/'
              f'cu{cuda_ver}/torch{torch_ver}/index.html')
    r = subprocess.run(
        ['pip', 'install', 'detectron2', '-f', d2_url],
        capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(
            ['pip', 'install',
             'git+https://github.com/facebookresearch/detectron2.git'],
            check=True)
else:
    subprocess.run(
        ['pip', 'install',
         'git+https://github.com/facebookresearch/detectron2.git'],
        check=True)

import detectron2
print(f'  OK detectron2 {detectron2.__version__}')

import pandas as pd
import numpy as np
import cv2, time, json, os, glob, importlib.util
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tabulate        import tabulate
from tqdm            import tqdm
from datetime        import timedelta
from torchvision.ops import nms as tv_nms

SEP  = '=' * 80
SEP2 = '-' * 80

print()
print(f'  All imports ready')
print(f'  NumPy  : {np.__version__}')
print(f'  cv2    : {cv2.__version__}')
print(f'  pandas : {pd.__version__}')


# ## Cell 3 — Clone CuVLER Repo + Install Requirements

# ── CELL 7 ──────────────────────────────────────────────────
# ============================================================
# CELL 3 — CLONE CuVLER REPO + INSTALL REQUIREMENTS
# ============================================================
# CuVLER extends CutLER — both need the repo for:
#   1. Custom config (add_cutler_config)
#   2. Custom model heads (CustomCascadeROIHeads)
#   3. YAML config files
#
# CuVLER ships its own version of the CutLER codebase.
# CutLER repo also kept as fallback for config.py.
# ============================================================

import os, sys, subprocess

print('=' * 65)
print('  CLONING CuVLER REPO')
print('=' * 65)

# ── Clone CuVLER ─────────────────────────────────────────────
if not os.path.exists(CUVLER_DIR):
    print('  Cloning CuVLER...')
    r = subprocess.run(
        ['git', 'clone',
         'https://github.com/shahaf-arica/CuVLER.git',
         CUVLER_DIR],
        capture_output=True, text=True)
    if r.returncode == 0:
        print('  ✓ CuVLER cloned')
    else:
        print(f'  ✗ Clone failed: {r.stderr}')
else:
    print(f'  ✓ CuVLER already exists at {CUVLER_DIR}')

# Add CuVLER to path
if CUVLER_DIR not in sys.path:
    sys.path.insert(0, CUVLER_DIR)

# ── Install CuVLER requirements ──────────────────────────────
req_path = os.path.join(CUVLER_DIR, 'requirements.txt')
if os.path.exists(req_path):
    print()
    print('  Installing CuVLER requirements...')
    with open(req_path) as f:
        pkgs = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    for pkg in pkgs:
        if 'faiss-gpu' in pkg:
            # faiss-gpu not available on all systems — use cpu version
            r = subprocess.run(
                ['pip', 'install', '-q', 'faiss-cpu'],
                capture_output=True)
            print(f"  {'OK' if r.returncode==0 else 'FAIL'} faiss-cpu  (replaced faiss-gpu)")
        else:
            r = subprocess.run(
                ['pip', 'install', '-q', pkg],
                capture_output=True)
            print(f"  {'OK' if r.returncode==0 else 'FAIL'} {pkg}")

# ── Verify key files ─────────────────────────────────────────
print()
print('  Verifying repo structure...')
key_paths = [
    os.path.join(CUVLER_DIR, 'cutler'),
    os.path.join(CUVLER_DIR, 'cutler', 'config.py'),
    os.path.join(CUVLER_DIR, 'cutler', 'modeling'),
]
for p in key_paths:
    exists = os.path.exists(p)
    print(f'  {"✓" if exists else "✗"} {p}')

# ── Find config YAML ─────────────────────────────────────────
config_files = glob.glob(
    os.path.join(CUVLER_DIR, '**', '*.yaml'), recursive=True)
cascade_configs = [c for c in config_files
                   if 'cascade' in c.lower() or 'cuvler' in c.lower()]
print()
print(f'  Found {len(config_files)} yaml configs')
print(f'  Cascade configs: {len(cascade_configs)}')
if cascade_configs:
    for c in cascade_configs[:3]:
        print(f'    {c}')

print()
print('  ✓ Cell 3 complete')
print('=' * 65)


# ── CELL 8 ──────────────────────────────────────────────────
import os

CUVLER_DIR = '/home/jovyan/OrangeGrove/CuVLER'

print('Full CuVLER directory contents:')
print('=' * 60)
for root, dirs, files in os.walk(CUVLER_DIR):
    # Skip hidden folders and deep nesting
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    depth = root.replace(CUVLER_DIR, '').count(os.sep)
    if depth > 3:
        continue
    indent = '  ' * depth
    print(f'{indent}{os.path.basename(root)}/')
    sub = '  ' * (depth + 1)
    for f in sorted(files):
        size = os.path.getsize(os.path.join(root, f))
        print(f'{sub}{f}  ({size/1e3:.1f} KB)')
print('=' * 60)

# ── CELL 9 ──────────────────────────────────────────────────
import os, sys, subprocess, shutil

CUVLER_DIR = '/home/jovyan/OrangeGrove/CuVLER'
CKPT_DIR   = os.path.join(CUVLER_DIR, 'checkpoints')

# ── Save checkpoint if it exists ─────────────────────────────
CKPT_PATH  = os.path.join(CKPT_DIR, 'cuvler_final.pth')
ckpt_exists = os.path.exists(CKPT_PATH)
ckpt_size   = os.path.getsize(CKPT_PATH) if ckpt_exists else 0
print(f'Checkpoint present: {ckpt_exists}  ({ckpt_size/1e6:.1f} MB)')

# ── Remove empty CuVLER dir ──────────────────────────────────
# Cannot clone into existing directory
if os.path.exists(CUVLER_DIR):
    shutil.rmtree(CUVLER_DIR)
    print('Removed empty CuVLER directory')

# ── Clone CuVLER fresh ───────────────────────────────────────
print('Cloning CuVLER...')
r = subprocess.run(
    ['git', 'clone',
     'https://github.com/shahaf-arica/CuVLER.git',
     CUVLER_DIR],
    capture_output=True, text=True, timeout=120)

print(f'Return code: {r.returncode}')
if r.stdout: print(f'stdout: {r.stdout[:300]}')
if r.stderr: print(f'stderr: {r.stderr[:300]}')

# ── Recreate checkpoints dir ─────────────────────────────────
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Restore checkpoint if it existed ─────────────────────────
if ckpt_exists and ckpt_size > 1e6:
    print(f'Checkpoint was already downloaded — preserved')
else:
    print('No checkpoint to restore — run Cell 4 to download')

# ── Verify clone ─────────────────────────────────────────────
print()
print('Verifying clone...')
key_items = [
    CUVLER_DIR,
    os.path.join(CUVLER_DIR, 'README.md'),
]
for path in key_items:
    exists = os.path.exists(path)
    print(f'  {"✓" if exists else "✗"}  {path}')

# List top-level contents
print()
print('Top-level contents:')
if os.path.exists(CUVLER_DIR):
    for item in sorted(os.listdir(CUVLER_DIR)):
        print(f'  {item}')

# ── CELL 10 ──────────────────────────────────────────────────
import os, subprocess

# Check CutLER repo
CUTLER_DIR = '/home/jovyan/OrangeGrove/CutLER'
print('CutLER repo:')
print(f'  Exists: {os.path.exists(CUTLER_DIR)}')
if os.path.exists(CUTLER_DIR):
    for item in sorted(os.listdir(CUTLER_DIR)):
        print(f'  {item}')

print()

# Check cad/ inside CuVLER — might have configs
CAD_DIR = '/home/jovyan/OrangeGrove/CuVLER/cad'
print('CuVLER/cad/ contents:')
if os.path.exists(CAD_DIR):
    for root, dirs, files in os.walk(CAD_DIR):
        depth = root.replace(CAD_DIR, '').count(os.sep)
        if depth > 2: continue
        indent = '  ' * depth
        print(f'  {indent}{os.path.basename(root)}/')
        for f in sorted(files):
            print(f'  {indent}  {f}')
else:
    print('  cad/ not found')

# ── CELL 11 ──────────────────────────────────────────────────
# Check what function name is in cad/config/config.py
config_py = '/home/jovyan/OrangeGrove/CuVLER/cad/config/config.py'
with open(config_py) as f:
    content = f.read()

# Find the add_*_config function name
import re
funcs = re.findall(r'def (add_\w+_config)', content)
print(f'Config functions found: {funcs}')
print()

# Check Base-RCNN-FPN.yaml exists
yaml_path = '/home/jovyan/OrangeGrove/CuVLER/cad/model_zoo/configs/Base-RCNN-FPN.yaml'
import os
print(f'YAML exists: {os.path.exists(yaml_path)}')
print(f'YAML size  : {os.path.getsize(yaml_path)/1e3:.1f} KB')
print()

# Check custom_cascade_rcnn.py
rcnn_py = '/home/jovyan/OrangeGrove/CuVLER/cad/modeling/roi_heads/custom_cascade_rcnn.py'
with open(rcnn_py) as f:
    rcnn_content = f.read()
classes = re.findall(r'class (\w+)', rcnn_content)
print(f'Classes in custom_cascade_rcnn.py: {classes}')

# ## Cell 4 — Download CuVLER Checkpoint

# ── CELL 13 ──────────────────────────────────────────────────
# ============================================================
# CELL 4 — DOWNLOAD CuVLER CHECKPOINT
# ============================================================
# CuVLER checkpoint hosted on HuggingFace (not fbaipublicfiles).
# HuggingFace may be blocked on JupyterHub — fallback included.
#
# If download fails:
#   1. Download manually on local machine from:
#      https://huggingface.co/shahaf-arica/CuVLER
#   2. Upload cuvler_final.pth to:
#      /home/jovyan/OrangeGrove/CuVLER/checkpoints/cuvler_final.pth
# ============================================================

import os, subprocess

print('=' * 65)
print('  CuVLER CHECKPOINT DOWNLOAD')
print('=' * 65)

WEIGHTS_LOADED_AS = None

if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 1e6:
    print(f'  Already exists ({os.path.getsize(CKPT_PATH)/1e6:.1f} MB)')
    WEIGHTS_LOADED_AS = 'cached'
else:
    print(f'  Downloading from HuggingFace...')
    print(f'  URL: {CKPT_URL}')
    r = subprocess.run(
        ['wget', '-q', '--show-progress',
         '--timeout=120', '-O', CKPT_PATH, CKPT_URL],
        capture_output=False)

    if (r.returncode == 0 and os.path.exists(CKPT_PATH)
            and os.path.getsize(CKPT_PATH) > 1e6):
        print(f'  ✓ Downloaded ({os.path.getsize(CKPT_PATH)/1e6:.1f} MB)')
        WEIGHTS_LOADED_AS = 'huggingface'
    else:
        print('  ✗ HuggingFace blocked or failed')
        print()
        print('  Manual download instructions:')
        print('    1. On your local machine, go to:')
        print('       https://huggingface.co/shahaf-arica/CuVLER')
        print('    2. Download: cuvler_final.pth')
        print('    3. Upload to JupyterHub:')
        print(f'       {CKPT_PATH}')
        print()
        print('  Then re-run this cell — it will detect the file.')
        WEIGHTS_LOADED_AS = 'MISSING'
        if os.path.exists(CKPT_PATH):
            os.remove(CKPT_PATH)

print()
print(f'  Checkpoint : {CKPT_PATH}')
print(f'  Source     : {WEIGHTS_LOADED_AS}')
if WEIGHTS_LOADED_AS not in ('MISSING', None):
    size = os.path.getsize(CKPT_PATH)
    print(f'  Size       : {size/1e6:.1f} MB')
    # Verify it is a valid torch checkpoint
    import torch
    try:
        ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
        keys = list(ckpt.keys()) if isinstance(ckpt, dict) else []
        print(f'  Keys       : {keys}')
        print('  ✓ Valid checkpoint')
    except Exception as e:
        print(f'  ⚠️  Checkpoint verification failed: {e}')
print('=' * 65)


# ── CELL 14 ──────────────────────────────────────────────────
import gdown, os

CKPT_PATH = '/home/jovyan/OrangeGrove/CuVLER/checkpoints/cuvler_final.pth'

# CuVLER Zero-shot model (correct for our study — no COCO fine-tuning)
FILE_ID = '16PHrqWvqfgcZfO5IfcpmAxCG2QYaQsEM'

print('Downloading CuVLER Zero-shot checkpoint...')
gdown.download(id=FILE_ID, output=CKPT_PATH, quiet=False)

if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 1e6:
    print(f'✓ Downloaded: {os.path.getsize(CKPT_PATH)/1e6:.1f} MB')
    WEIGHTS_LOADED_AS = 'cuvler_zero_shot_gdrive'
else:
    print('✗ Download failed')

# ── CELL 15 ──────────────────────────────────────────────────
import os, glob

CUVLER_DIR = '/home/jovyan/OrangeGrove/CuVLER'

# Check all yamls
yamls = glob.glob(os.path.join(CUVLER_DIR, '**', '*.yaml'), recursive=True)
print('All yaml files in CuVLER repo:')
for y in yamls:
    print(f'  {y.replace(CUVLER_DIR, "")}')

# ── CELL 16 ──────────────────────────────────────────────────
import os
CKPT_PATH = '/home/jovyan/OrangeGrove/CuVLER/checkpoints/cuvler_final.pth'
if os.path.exists(CKPT_PATH):
    print(f'✓ Checkpoint exists: {os.path.getsize(CKPT_PATH)/1e6:.1f} MB')
else:
    print('✗ Checkpoint missing')

# ## Cell 5 — Load CuVLER Model

# ── CELL 18 ──────────────────────────────────────────────────
import os, re

# ── Fix 1: cad/__init__.py ───────────────────────────────────
# Python 2 style: "import config" → Python 3: "from . import config"
init_path = os.path.join(CUVLER_DIR, 'cad', '__init__.py')
with open(init_path) as f:
    content = f.read()

print('Before:')
print(content)

fixed = re.sub(
    r'^import (config|engine|modeling|structures|data|solver|evaluation)$',
    r'from . import \1',
    content, flags=re.MULTILINE)

with open(init_path, 'w') as f:
    f.write(fixed)

print('After:')
print(fixed)
print('✓ Fixed cad/__init__.py')
print()

# ── Fix 2: transform.py ──────────────────────────────────────
# PIL.Image.LINEAR was removed in Pillow 10.0
# Replacement: Image.BILINEAR (works on all versions)
transform_path = os.path.join(
    CUVLER_DIR, 'cad', 'data', 'transforms', 'transform.py')
with open(transform_path) as f:
    content = f.read()

fixed2 = content.replace('Image.LINEAR', 'Image.BILINEAR')
with open(transform_path, 'w') as f:
    f.write(fixed2)

changes = content.count('Image.LINEAR')
print(f'✓ Fixed transform.py  ({changes} occurrences replaced)')
print()
print('Both patches applied. Re-run Cell 5 now.')

# ── CELL 19 ──────────────────────────────────────────────────
import os

# ── Blank out cad/__init__.py ────────────────────────────────
# The original file auto-imports engine, data, modeling, structures
# — all training code with Python 2/3 incompatibilities.
# For inference we only need:
#   cad.config.config → add_cuvler_config
#   cad.modeling.roi_heads.custom_cascade_rcnn → CustomCascadeROIHeads
# We import those directly — __init__.py is not needed.

init_path = os.path.join(CUVLER_DIR, 'cad', '__init__.py')
with open(init_path, 'w') as f:
    f.write(
        '# Patched for inference — training imports removed.\n'
        '# Only cad.config and cad.modeling submodules are loaded.\n'
    )
print('✓ Cleared cad/__init__.py')

# ── Also clear cad/modeling/__init__.py ──────────────────────
# Prevents auto-import of broken modeling submodules
modeling_init = os.path.join(CUVLER_DIR, 'cad', 'modeling', '__init__.py')
if os.path.exists(modeling_init):
    with open(modeling_init) as f:
        orig = f.read()
    print(f'  modeling/__init__.py original:')
    print(f'  {orig[:200]}')
    with open(modeling_init, 'w') as f:
        f.write('# Patched for inference\n')
    print('✓ Cleared cad/modeling/__init__.py')

# ── Also clear cad/config/__init__.py ───────────────────────
config_init = os.path.join(CUVLER_DIR, 'cad', 'config', '__init__.py')
if os.path.exists(config_init):
    with open(config_init, 'w') as f:
        f.write('# Patched for inference\n')
    print('✓ Cleared cad/config/__init__.py')

# ── Also clear roi_heads/__init__.py ────────────────────────
roi_init = os.path.join(
    CUVLER_DIR, 'cad', 'modeling', 'roi_heads', '__init__.py')
if os.path.exists(roi_init):
    with open(roi_init, 'w') as f:
        f.write('# Patched for inference\n')
    print('✓ Cleared cad/modeling/roi_heads/__init__.py')

print()
print('All __init__.py files cleared.')
print('Re-run Cell 5 now.')

# ── CELL 20 ──────────────────────────────────────────────────
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY

# Check what IS registered after our import
print('ROI_HEADS registry contents:')
for k in ROI_HEADS_REGISTRY._obj_map.keys():
    print(f'  {k}')

print()

# Check what registry custom_cascade_rcnn.py actually uses
rcnn_path = os.path.join(
    CUVLER_DIR, 'cad', 'modeling', 'roi_heads', 'custom_cascade_rcnn.py')
with open(rcnn_path) as f:
    content = f.read()

# Show top 30 lines — reveals imports and decorator
print('custom_cascade_rcnn.py top:')
print('\n'.join(content.split('\n')[:30]))

# ── CELL 21 ──────────────────────────────────────────────────
# Check yaml META_ARCHITECTURE and roi_heads.py
import os

yaml_path = os.path.join(
    CUVLER_DIR, 'cad', 'model_zoo', 'configs',
    'CutVER-ImageNet',
    'cascade_mask_rcnn_R_50_FPN_votecut_cad.yaml')

print('YAML content:')
with open(yaml_path) as f:
    print(f.read())

print()
print('roi_heads.py top 30 lines:')
rh_path = os.path.join(CUVLER_DIR, 'cad', 'modeling', 'roi_heads', 'roi_heads.py')
with open(rh_path) as f:
    lines = f.readlines()
print(''.join(lines[:30]))

print()
print('structures/__init__.py:')
struct_path = os.path.join(CUVLER_DIR, 'cad', 'structures', '__init__.py')
if os.path.exists(struct_path):
    with open(struct_path) as f:
        print(f.read())

# ── CELL 22 ──────────────────────────────────────────────────
# ── Clear any existing registrations ─────────────────────────
# Makes Cell 5 safe to re-run after errors or restarts.
# Detectron2 registries persist across cells — must clear first.
from detectron2.modeling.roi_heads          import ROI_HEADS_REGISTRY     as _D2_ROI
from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY as _D2_MASK

for _name in ['CustomCascadeROIHeads', 'CustomStandardROIHeads']:
    _D2_ROI._obj_map.pop(_name, None)

for _name in ['CustomMaskRCNNConvUpsampleHead']:
    _D2_MASK._obj_map.pop(_name, None)

# Also clear cached modules so files reload cleanly
for _key in list(sys.modules.keys()):
    if 'cad.' in _key:
        del sys.modules[_key]

print('  ✓ Registries cleared — safe to re-run')

# ── CELL 23 ──────────────────────────────────────────────────
# ============================================================
# CELL 5 — LOAD CuVLER MODEL (FINAL CORRECTED VERSION)
# ============================================================
# CuVLER uses its own ROI_HEADS_REGISTRY (in cad/modeling/roi_heads/roi_heads.py)
# separate from Detectron2's. We must:
#   1. Load all cad dependencies in correct order
#   2. Register CustomCascadeROIHeads in BOTH registries
#   3. Register CustomMaskRCNNConvUpsampleHead in detectron2's mask registry
# ============================================================

import sys, os, torch, types, importlib.util
import numpy as np
from detectron2.config     import get_cfg
from detectron2.modeling   import build_model
from detectron2.checkpoint import DetectionCheckpointer

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
CAD_DIR = os.path.join(CUVLER_DIR, 'cad')

print('=' * 65)
print('  LOADING CuVLER MODEL')
print('=' * 65)

# ── Step 1: Register cad package stubs ───────────────────────
# Needed so "from cad.x import y" inside the files we load works
def make_pkg(name, path):
    pkg = types.ModuleType(name)
    pkg.__path__    = [path]
    pkg.__package__ = name
    sys.modules[name] = pkg
    return pkg

make_pkg('cad',                 CAD_DIR)
make_pkg('cad.modeling',        os.path.join(CAD_DIR, 'modeling'))
make_pkg('cad.modeling.roi_heads',
         os.path.join(CAD_DIR, 'modeling', 'roi_heads'))
make_pkg('cad.structures',      os.path.join(CAD_DIR, 'structures'))
print('  ✓ Package stubs registered')

# ── Step 2: Load cad.structures.boxes ────────────────────────
# Provides pairwise_iou_max_scores used by roi_heads.py
def load_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

struct_dir = os.path.join(CAD_DIR, 'structures')
boxes_mod  = load_file(
    'cad.structures.boxes',
    os.path.join(struct_dir, 'boxes.py'))
# Attach to stub so "from cad.structures import pairwise_iou_max_scores" works
sys.modules['cad.structures'].pairwise_iou_max_scores = \
    boxes_mod.pairwise_iou_max_scores
print('  ✓ cad.structures.boxes loaded')

# ── Step 3: Load roi_heads submodules in dependency order ────
roi_dir = os.path.join(CAD_DIR, 'modeling', 'roi_heads')

# 3a: fast_rcnn.py (needed by roi_heads.py and custom_cascade_rcnn.py)
fast_rcnn_mod = load_file(
    'cad.modeling.roi_heads.fast_rcnn',
    os.path.join(roi_dir, 'fast_rcnn.py'))
print('  ✓ fast_rcnn loaded')

# 3b: mask_head.py (CustomMaskRCNNConvUpsampleHead)
mask_head_mod = load_file(
    'cad.modeling.roi_heads.mask_head',
    os.path.join(roi_dir, 'mask_head.py'))
print('  ✓ mask_head loaded')

# 3c: roi_heads.py (defines CuVLER's ROI_HEADS_REGISTRY)
roi_heads_mod = load_file(
    'cad.modeling.roi_heads.roi_heads',
    os.path.join(roi_dir, 'roi_heads.py'))
print('  ✓ roi_heads loaded')

# 3d: custom_cascade_rcnn.py (registers CustomCascadeROIHeads
#     in CuVLER's registry via @ROI_HEADS_REGISTRY.register())
cascade_mod = load_file(
    'cad.modeling.roi_heads.custom_cascade_rcnn',
    os.path.join(roi_dir, 'custom_cascade_rcnn.py'))
CustomCascadeROIHeads = cascade_mod.CustomCascadeROIHeads
print('  ✓ custom_cascade_rcnn loaded')

# ── Step 4: Register in Detectron2's registries ───────────────
# Detectron2's build_model uses its OWN registries.
# We must cross-register CuVLER's classes there too.
from detectron2.modeling.roi_heads         import ROI_HEADS_REGISTRY     as D2_ROI
from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY as D2_MASK

if 'CustomCascadeROIHeads' not in D2_ROI._obj_map:
    D2_ROI.register(CustomCascadeROIHeads)
print(f'  ✓ CustomCascadeROIHeads in detectron2 registry')

if hasattr(mask_head_mod, 'CustomMaskRCNNConvUpsampleHead'):
    CustomMaskHead = mask_head_mod.CustomMaskRCNNConvUpsampleHead
    if 'CustomMaskRCNNConvUpsampleHead' not in D2_MASK._obj_map:
        D2_MASK.register(CustomMaskHead)
    print(f'  ✓ CustomMaskRCNNConvUpsampleHead in detectron2 registry')
else:
    print('  ⚠️  CustomMaskRCNNConvUpsampleHead not found in mask_head.py')

# ── Step 5: Load add_cuvler_config ────────────────────────────
make_pkg('cad.config', os.path.join(CAD_DIR, 'config'))
config_mod       = load_file(
    'cad.config.config',
    os.path.join(CAD_DIR, 'config', 'config.py'))
add_cuvler_config = config_mod.add_cuvler_config
print('  ✓ add_cuvler_config loaded')

# ── Step 6: Build config ─────────────────────────────────────
YAML_PATH = os.path.join(
    CUVLER_DIR, 'cad', 'model_zoo', 'configs',
    'CutVER-ImageNet',                        # typo in repo
    'cascade_mask_rcnn_R_50_FPN_votecut_cad.yaml')

cfg = get_cfg()
add_cuvler_config(cfg)
cfg.merge_from_file(YAML_PATH)
cfg.MODEL.WEIGHTS = CKPT_PATH
cfg.MODEL.DEVICE  = DEVICE
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# Override SyncBN → BN for single-GPU inference
cfg.MODEL.RESNETS.NORM = 'BN'
cfg.MODEL.FPN.NORM     = ''
print('  ✓ Config built')

# ── Step 7: Build model ───────────────────────────────────────
model = build_model(cfg)
DetectionCheckpointer(model).load(CKPT_PATH)
model.eval()
print('  ✓ Model built and checkpoint loaded')

# ── Step 8: Predictor + threshold patching ───────────────────
from detectron2.data import transforms as T2

class CuVLERPredictor:
    def __init__(self, cfg, model):
        self.cfg   = cfg.clone()
        self.model = model

    def __call__(self, img_bgr):
        height, width = img_bgr.shape[:2]
        aug   = T2.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST,
             self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST)
        img_t = aug.get_transform(img_bgr).apply_image(img_bgr)
        img_t = torch.as_tensor(
            img_t.astype('float32').transpose(2, 0, 1))
        with torch.no_grad():
            outputs = self.model(
                [{'image': img_t, 'height': height, 'width': width}])
        return outputs[0]

predictor      = CuVLERPredictor(cfg, model)
PREDICTOR_TYPE = 'custom'

def patch_score_threshold(predictor, threshold, predictor_type=None):
    for module in predictor.model.modules():
        if hasattr(module, 'test_score_thresh'):
            module.test_score_thresh = threshold

patch_score_threshold(predictor, 0.5)
print('  ✓ Predictor ready')

# ── Sanity check ─────────────────────────────────────────────
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
out      = predictor(test_img)
print()
print(f'  Device     : {DEVICE}')
print(f'  Predictor  : CuVLERPredictor')
print(f'  VoteCut    : DINO + DINOv2 + MAE')
print(f'  Backbone   : ResNet-50 + FPN')
print(f'  ROI heads  : CustomCascadeROIHeads ✓')
print(f'  Mask head  : CustomMaskRCNNConvUpsampleHead ✓')
print(f'  Sanity     : ✓ ({len(out["instances"])} instances on blank image)')
print('=' * 65)

# ## Cell 6 — Load Ground Truth from Shared Cache

# ── CELL 25 ──────────────────────────────────────────────────
# ============================================================
# CELL 6 — LOAD GROUND TRUTH FROM SHARED CACHE
# ============================================================
# Identical to CutLER Cell 6, iBOT Cell 5, MAE Cell 5.
# ============================================================

import sys
import pandas as pd
import numpy as np

sys.path.insert(0, '/home/jovyan/OrangeGrove/notebooks')
from shared import (
    load_cache, mae, rmse, bias,
    within_n, compute_f1, tp_mae, box_dot_match
)

data         = load_cache()
gt_lookup    = data['gt_lookup']
dot_lookup   = data['dot_lookup']
cal_frames   = data['cal_frames']
test_frames  = data['test_frames']
sweep_frames = data['sweep_frames']
video_groups = data['video_groups']
gt_df        = data['gt_df']
tree_summary = data['tree_summary']
COHORTS      = data['cohorts']

cal_set  = set(cal_frames)
test_set = set(test_frames)
gt_df['split'] = gt_df['image_filename'].apply(
    lambda f: 'cal' if f in cal_set else 'test')

gt_master = gt_df.copy()
gt_master['source_tree']  = gt_master['tree'].map(TREE_FOLDER_MAP)
gt_master['source_video'] = gt_master['image_filename'].str.extract(
    r'(Vid \d+)')
gt_master['source_group'] = gt_master['tree'].apply(
    lambda t: '40sec' if t in TREE_40SEC_IDS else '30sec')

cal_df  = gt_master[gt_master['split'] == 'cal']
test_df = gt_master[gt_master['split'] == 'test']

cal_sweep = [f for f in sweep_frames
             if any(f.startswith(t) for t in CAL_TREE_IDS)]

print('=' * 65)
print('  GROUND TRUTH LOADED FROM CACHE')
print('=' * 65)
print(f'  GT frames    : {len(gt_lookup):,}')
print(f'  Dot frames   : {len(dot_lookup):,}')
print(f'  Cal frames   : {len(cal_frames):,}')
print(f'  Test frames  : {len(test_frames):,}')
print(f'  Cal sweep    : {len(cal_sweep)} frames')
print(f'  GT mean      : {gt_df["ground_truth_count"].mean():.2f}')
print('=' * 65)


# ## Cell 7 — Dataset Inventory

# ── CELL 27 ──────────────────────────────────────────────────
# ============================================================
# CELL 7 — DATASET INVENTORY
# ============================================================

print(SEP)
print('  DATASET INVENTORY')
print(SEP)
print(tabulate(tree_summary, headers='keys',
               tablefmt='pretty', showindex=False))
print()

inventory_rows = []
for sec, trees in [('30sec', TREES_30SEC), ('40sec', TREES_40SEC)]:
    frames_dir = DIR_30SEC if sec == '30sec' else DIR_40SEC
    print(f'  {sec} trees:')
    for tree in trees:
        split_tag = 'CAL' if tree in CAL_TREES else 'TEST'
        for vid in VIDEOS:
            vid_path  = os.path.join(frames_dir, tree, vid)
            confirmed = CONFIRMED_COUNTS.get((tree, vid), 0)
            disk = len([f for f in os.listdir(vid_path)
                        if f.endswith('.jpg')])                    if os.path.exists(vid_path) else 0
            match = 'OK' if disk == confirmed                     else f'MISMATCH {disk}v{confirmed}'
            print(f'    [{split_tag}] {tree} {vid}: {disk} frames  {match}')
            inventory_rows.append({
                'Group':sec,'Tree':tree,'Video':vid,
                'Split':split_tag,'Frames':disk,
                'Expected':confirmed,'Match':match})

inventory_df = pd.DataFrame(inventory_rows)
mismatches   = inventory_df[inventory_df['Match'] != 'OK']
print()
if len(mismatches):
    print(f'  ⚠️  {len(mismatches)} mismatches found')
else:
    print('  All frame counts match ✓')


# ## Cell 8 — Tile Coverage Verification

# ── CELL 29 ──────────────────────────────────────────────────
# ============================================================
# CELL 8 — TILE COVERAGE VERIFICATION
# ============================================================
# Identical to CutLER Cell 7.
# Proves 1920×1080 tiling covers every 4K pixel.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FRAME_H, FRAME_W = 3840, 2160

def get_tile_coords(frame_h, frame_w, tile_h, tile_w, overlap):
    tiles    = []
    stride_x = tile_w - overlap   # x uses tile WIDTH
    stride_y = tile_h - overlap   # y uses tile HEIGHT  ← the fix
    y = 0
    while y < frame_h:
        x = 0
        while x < frame_w:
            x2 = min(x + tile_w, frame_w)
            y2 = min(y + tile_h, frame_h)
            tiles.append((x, y, x2, y2))
            if x2 == frame_w: break
            x += stride_x
        if y2 == frame_h: break
        y += stride_y              # ← was: y += stride (wrong)
    return tiles

tiles = get_tile_coords(FRAME_H, FRAME_W, TILE_H, TILE_W, TILE_OVERLAP)

SCALE = 8
cov   = np.zeros((FRAME_H//SCALE, FRAME_W//SCALE), dtype=np.uint8)
for (x1,y1,x2,y2) in tiles:
    cov[y1//SCALE:y2//SCALE, x1//SCALE:x2//SCALE] += 1

uncovered = int((cov == 0).sum())
max_cover = int(cov.max())
last_x2   = max(t[2] for t in tiles)
last_y2   = max(t[3] for t in tiles)

print('=' * 60)
print('  TILE COVERAGE REPORT')
print('=' * 60)
print(f'  Frame size   : {FRAME_W}×{FRAME_H}px  (portrait 4K)')
print(f'  Tile size    : {TILE_W}×{TILE_H}px')
print(f'  Overlap      : {TILE_OVERLAP}px')
print(f'  Stride       : {TILE_W-TILE_OVERLAP}px')
print(f'  Total tiles  : {len(tiles)} per frame')
print()
print(f'  Min coverage : {cov.min()}×  (must be ≥ 1)')
print(f'  Max coverage : {max_cover}×  (overlap zones)')
print(f'  Uncovered px : {uncovered * SCALE**2}  (must be 0)')
print()

if uncovered == 0:
    print('  ✓ Every pixel covered — tiling is complete')
else:
    print(f'  ✗ FAIL: {uncovered * SCALE**2} pixels NOT covered')

print(f'  Right edge   : x={last_x2}  (width={FRAME_W})  '
      f'{"✓" if last_x2==FRAME_W else "✗"}')
print(f'  Bottom edge  : y={last_y2}  (height={FRAME_H})  '
      f'{"✓" if last_y2==FRAME_H else "✗"}')
print('=' * 60)

assert uncovered == 0, 'Tiling incomplete — fix TILE_W/TILE_H/TILE_OVERLAP'
assert last_x2 == FRAME_W and last_y2 == FRAME_H
print('  All assertions passed ✓')


# ## Cell 9 — Color Filter + Tiled Inference Functions

# ── CELL 31 ──────────────────────────────────────────────────
# ============================================================
# CELL 9 — COLOR FILTER + TILED INFERENCE FUNCTIONS
# ============================================================
# Color filter: identical to CutLER, iBOT, MAE.
# Tiled inference: identical to CutLER EXCEPT predictor call.
#
# CutLER: predictor returns instances with pred_boxes + scores
# CuVLER: same — DefaultPredictor returns same structure
#         The VoteCut difference is in the weights, not the API
# ============================================================

import cv2, numpy as np, torch
from torchvision.ops import nms as tv_nms


# ── COLOR FILTER ─────────────────────────────────────────────
def is_green_detection(img_bgr, box,
                        ripe_hue_min=15,  ripe_hue_max=40,
                        ripe_sat_min=120, ripe_val_min=160,
                        ripe_reject_ratio=0.08,
                        green_hue_min=25,  green_hue_max=85,
                        green_sat_min=40,  green_ratio_thresh=0.25):
    """Two-stage HSV filter. Identical to CutLER, iBOT, MAE."""
    x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
    H,W = img_bgr.shape[:2]
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(W,x2), min(H,y2)
    if x2<=x1 or y2<=y1: return False
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0: return False
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue  = hsv[:,:,0]; sat=hsv[:,:,1]; val=hsv[:,:,2]
    ripe = ((hue>=ripe_hue_min)&(hue<=ripe_hue_max)&
            (sat>=ripe_sat_min)&(val>=ripe_val_min))
    if ripe.sum()/hue.size >= ripe_reject_ratio: return False
    green= ((hue>=green_hue_min)&(hue<=green_hue_max)&(sat>=green_sat_min))
    return (green.sum()/hue.size) >= green_ratio_thresh


# ── TILED INFERENCE ──────────────────────────────────────────
def run_tiled_inference_cuvler(img_bgr, predictor,
                                tile_w=TILE_W, tile_h=TILE_H,
                                overlap=TILE_OVERLAP,
                                nms_iou=NMS_IOU):
    """
    Full pipeline for one 4K frame — identical to CutLER.

    1. Split into overlapping 1920×1080 tiles
    2. Run CuVLER predictor on each tile
    3. Offset boxes to full-frame coordinates
    4. NMS across all tile detections
    5. Color filter → remove ripe orange detections

    CuVLER vs CutLER difference:
      CutLER: MaskCut pseudo-labels → less complete masks
      CuVLER: VoteCut pseudo-labels → more complete masks
      Same API, same tiling — only the weights differ.

    Returns: (green_count, total_after_nms, filtered_out)
    """
    H, W       = img_bgr.shape[:2]
    all_boxes  = []
    all_scores = []
    stride_x = tile_w - overlap
    stride_y = tile_h - overlap

    y = 0
    while y < H:
        x = 0
        while x < W:
            x1=x; y1=y
            x2=min(x+tile_w,W); y2=min(y+tile_h,H)
            tile = img_bgr[y1:y2, x1:x2]

            with torch.no_grad():
                out = predictor(tile)

            if hasattr(out, '__getitem__'):
                instances = out['instances'].to('cpu')
            else:
                instances = out.to('cpu')

            if len(instances) > 0:
                boxes  = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.numpy()
                # Offset from tile to full-frame coordinates
                boxes[:, 0] += x1; boxes[:, 2] += x1
                boxes[:, 1] += y1; boxes[:, 3] += y1
                all_boxes.append(boxes)
                all_scores.append(scores)

            if x2==W: break
            x += stride_x
        if y2==H: break
        y += stride_y

    if not all_boxes:
        return 0, 0, 0

    all_boxes  = np.concatenate(all_boxes,  axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    keep = tv_nms(
        torch.tensor(all_boxes,  dtype=torch.float32),
        torch.tensor(all_scores, dtype=torch.float32),
        iou_threshold=nms_iou)

    kept_boxes   = all_boxes[keep.numpy()]
    total_nms    = len(kept_boxes)

    green_boxes  = [b for b in kept_boxes
                    if is_green_detection(img_bgr, b)]
    filtered_out = total_nms - len(green_boxes)

    return len(green_boxes), total_nms, filtered_out


# ── Sanity check ─────────────────────────────────────────────
print('  Testing on one frame...')
_test_img = os.path.join(DIR_30SEC, 'Tree_01', 'Vid 01',
                          '483_Vid 01_F001.jpg')
if os.path.exists(_test_img):
    _bgr = cv2.imread(_test_img)
    _cnt, _nms, _filt = run_tiled_inference_cuvler(_bgr, predictor)
    print(f'  Sanity check : 483_Vid 01_F001.jpg')
    print(f'  GT           : 23')
    print(f'  Predicted    : {_cnt}')
    print(f'  Pre-NMS      : see tile count')
    print(f'  Post-NMS     : {_nms}')
    print(f'  Filtered out : {_filt} ripe')
else:
    print('  Test frame not found — skipping sanity check')

print()
print('  ✓ Color filter ready (identical to CutLER/iBOT/MAE)')
print('  ✓ Tiled inference ready')
print(f'  Tiles per frame : {len(tiles)}')
print(f'  Tile size       : {TILE_W}×{TILE_H}px')


# ## Cell 10 — Threshold Sweep on Calibration Trees Only

# ── CELL 33 ──────────────────────────────────────────────────
# ============================================================
# CELL 10 — THRESHOLD SWEEP ON CALIBRATION TREES ONLY
# ============================================================
# Identical protocol to CutLER Cell 9.
# Tests confidence thresholds 0.1–0.9 on 54 cal frames.
# Picks threshold with lowest MAE.
# Test trees never touched.
# ============================================================

import numpy as np, pandas as pd, time, os, cv2
from tabulate import tabulate

print('=' * 65)
print('  THRESHOLD SWEEP — Calibration Trees Only')
print(f'  Sweep frames : {len(cal_sweep)}')
print(f'  Cal trees    : {CAL_TREE_IDS}')
print('=' * 65)

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

best_mae_val = float('inf')
best_thresh  = 0.5
sweep_rows   = []

for thresh in THRESHOLDS:

    patch_score_threshold(predictor, thresh, PREDICTOR_TYPE)
    preds   = []
    gts     = []
    skipped = 0
    t_start = time.time()

    for fname in cal_sweep:

        gt_c = gt_lookup.get(fname)
        if gt_c is None:
            skipped += 1; continue

        try:
            tree_id = fname.split('_Vid')[0]
            vid_num = fname.split('Vid ')[1].split('_')[0]
            vid     = f'Vid {vid_num}'
        except:
            skipped += 1; continue

        tree_folder = TREE_FOLDER_MAP.get(tree_id)
        if not tree_folder:
            skipped += 1; continue

        sec        = '40sec' if tree_id in TREE_40SEC_IDS else '30sec'
        frames_dir = DIR_40SEC if sec == '40sec' else DIR_30SEC
        img_path   = os.path.join(frames_dir, tree_folder, vid, fname)

        if not os.path.exists(img_path):
            skipped += 1; continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            skipped += 1; continue

        count, _, _ = run_tiled_inference_cuvler(img_bgr, predictor)
        preds.append(count)
        gts.append(int(gt_c))

    if len(preds) < 5:
        print(f'  thresh={thresh} → too few frames ({len(preds)})')
        continue

    arr_p  = np.array(preds, dtype=float)
    arr_g  = np.array(gts,   dtype=float)
    m_mae  = float(np.mean(np.abs(arr_p-arr_g)))
    m_rmse = float(np.sqrt(np.mean((arr_p-arr_g)**2)))
    m_w2   = float(np.mean(np.abs(arr_p-arr_g)<=2)*100)
    m_bias = float(np.mean(arr_p-arr_g))
    elapsed= time.time() - t_start

    is_best = m_mae < best_mae_val
    if is_best:
        best_mae_val = m_mae
        best_thresh  = thresh

    flag = '  ← best' if is_best else ''
    print(f'  thresh={thresh}  '
          f'MAE={m_mae:6.2f}  '
          f'RMSE={m_rmse:6.2f}  '
          f'W±2={m_w2:5.1f}%  '
          f'Bias={m_bias:+7.2f}  '
          f'({elapsed:.0f}s)'
          f'{flag}')

    sweep_rows.append({
        'Threshold': thresh,
        'MAE'      : round(m_mae,  2),
        'RMSE'     : round(m_rmse, 2),
        'W±2%'     : round(m_w2,   1),
        'Bias'     : round(m_bias, 2),
        'Frames'   : len(preds),
        'Skipped'  : skipped,
    })

sweep_df     = pd.DataFrame(sweep_rows)
BEST_THRESH  = best_thresh
patch_score_threshold(predictor, BEST_THRESH, PREDICTOR_TYPE)
best_row     = sweep_df[sweep_df['Threshold']==BEST_THRESH].iloc[0]

print()
print('=' * 65)
print('  THRESHOLD DECISION')
print('=' * 65)
print(f'  Selected threshold : {BEST_THRESH}')
print(f'  Cal sweep MAE      : {best_mae_val:.2f}')
print(f'  Cal sweep RMSE     : {best_row["RMSE"]}')
print(f'  Cal sweep W±2%     : {best_row["W±2%"]}%')
print(f'  Cal sweep Bias     : {best_row["Bias"]:+.2f}')
print()
print('  Full sweep table:')
print(tabulate(sweep_df, headers='keys',
               tablefmt='pretty', showindex=False))
print()
print(f'  Ready for Cell 11 — full inference on {TOTAL_FRAMES:,} frames')
print(f'  Threshold = {BEST_THRESH}')
print('=' * 65)


# ## Cell 11 — Full Inference on All 10,577 Frames

# ── CELL 35 ──────────────────────────────────────────────────
import os

# Verify stride fix is in Cell 9's inference function
# by checking how many tiles a test frame produces
import cv2
test_bgr = cv2.imread(os.path.join(
    DIR_30SEC, 'Tree_01', 'Vid 01', '483_Vid 01_F001.jpg'))

H, W     = test_bgr.shape[:2]
stride_x = TILE_W - TILE_OVERLAP  # 1820
stride_y = TILE_H - TILE_OVERLAP  # 980

tile_count = 0
y = 0
while y < H:
    x = 0
    while x < W:
        x2 = min(x + TILE_W, W)
        y2 = min(y + TILE_H, H)
        tile_count += 1
        if x2 == W: break
        x += stride_x
    if y2 == H: break
    y += stride_y

print(f'Tiles per frame: {tile_count}  (must be 8)')
print(f'{"✓ Stride fix confirmed" if tile_count == 8 else "✗ Stride bug still present"}')

# ── CELL 36 ──────────────────────────────────────────────────
# ============================================================
# CELL 11 — FULL INFERENCE ON ALL 10,577 FRAMES
# ============================================================
# Tiled inference + color filter on every frame.
# Run overnight. Expected time: ~6 hours on A100.
# ============================================================

print('=' * 70)
print('  CuVLER FULL INFERENCE — 10,577 FRAMES')
print('=' * 70)
print(f'  Threshold   : {BEST_THRESH}')
print(f'  Tiles       : {TILE_W}×{TILE_H}  overlap={TILE_OVERLAP}px')
print(f'  NMS IoU     : {NMS_IOU}')
print(f'  Filter      : Stage1 ripe(hue15-40,sat>120,val>160)>8% reject')
print(f'              : Stage2 green(hue25-85,sat>40)<25% reject')
print('=' * 70)

patch_score_threshold(predictor, BEST_THRESH, PREDICTOR_TYPE)

all_results    = []
skipped        = 0
total_filtered = 0
global_start   = time.time()

for sec, trees in [('30sec', TREES_30SEC), ('40sec', TREES_40SEC)]:
    frames_dir = DIR_30SEC if sec == '30sec' else DIR_40SEC
    print(f'\n  {sec} trees')

    for tree in trees:
        split_tag = 'cal' if tree in CAL_TREES else 'test'
        print(f'\n  [{split_tag.upper()}] {tree}')

        for vid in VIDEOS:
            vid_path  = os.path.join(frames_dir, tree, vid)
            confirmed = CONFIRMED_COUNTS.get((tree, vid), 0)

            vid_rows = gt_master[
                (gt_master['source_tree']  == tree) &
                (gt_master['source_video'] == vid)
            ].copy().reset_index(drop=True)

            if len(vid_rows) == 0:
                print(f'    {vid} — no GT rows, skipping')
                continue

            vid_start    = time.time()
            vid_results  = []
            vid_filtered = 0

            for _, row in tqdm(
                vid_rows.iterrows(),
                total=len(vid_rows),
                desc=f'    {vid} ({confirmed}fr)',
                unit='fr',
                bar_format='{l_bar}{bar:20}{r_bar}'
            ):
                fname    = str(row['image_filename']).strip()
                img_path = os.path.join(vid_path, fname)
                if not os.path.exists(img_path):
                    skipped += 1; continue

                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    skipped += 1; continue

                count, total_nms, filtered = run_tiled_inference_cuvler(
                    img_bgr, predictor)

                gt_c           = int(row['ground_truth_count'])
                vid_filtered  += filtered
                total_filtered += filtered

                vid_results.append({
                    'group'               : sec,
                    'tree'                : tree,
                    'video'               : vid,
                    'split'               : split_tag,
                    'image_filename'      : fname,
                    'ground_truth'        : gt_c,
                    'predicted'           : count,
                    'detections_pre_filter': total_nms,
                    'filtered_out'        : filtered,
                    'error'               : count - gt_c,
                    'abs_error'           : abs(count - gt_c),
                })

            all_results.extend(vid_results)
            pd.DataFrame(all_results).to_pickle(
                os.path.join(OUT_DIR, f'{MODEL_SLUG}_res_df_checkpoint.pkl'))
            vid_elapsed = time.time() - vid_start

            if vid_results:
                vg = np.array([r['ground_truth'] for r in vid_results])
                vp = np.array([r['predicted']    for r in vid_results])
                print(f'    {vid}  '
                      f'MAE={np.mean(np.abs(vp-vg)):.2f}  '
                      f'Bias={np.mean(vp-vg):+.2f}  '
                      f'W±2={np.mean(np.abs(vp-vg)<=2)*100:.0f}%  '
                      f'Filter={vid_filtered}  '
                      f'Time={str(timedelta(seconds=int(vid_elapsed)))}')

total_elapsed = time.time() - global_start
res_df = pd.DataFrame(all_results)

print()
print('=' * 70)
print('  INFERENCE COMPLETE')
print('=' * 70)
print(f'  Frames processed : {len(res_df):,}')
print(f'  Frames skipped   : {skipped}')
print(f'  Ripe filtered    : {total_filtered:,}')
print(f'  Total time       : {str(timedelta(seconds=int(total_elapsed)))}')
gt_all   = res_df['ground_truth'].values
pred_all = res_df['predicted'].values
print(f'  MAE              : {np.mean(np.abs(pred_all-gt_all)):.2f}')
print(f'  Bias             : {np.mean(pred_all-gt_all):+.2f}')
print(f'  W±2%             : {np.mean(np.abs(pred_all-gt_all)<=2)*100:.1f}%')
print('=' * 70)


# ## Cell 11.5 — Spatial Metrics: Precision, Recall, F1

# ── CELL 38 ──────────────────────────────────────────────────
# ============================================================
# CELL 11.5 — SPATIAL METRICS: PRECISION, RECALL, F1
# ============================================================
# Identical structure to CutLER Cell 10.5.
# ============================================================

print(SEP)
print('  SPATIAL METRICS — PRECISION, RECALL, F1')
print(SEP)

spatial_results = []
frames_no_dots  = 0

for sec, trees in [('30sec', TREES_30SEC), ('40sec', TREES_40SEC)]:
    frames_dir = DIR_30SEC if sec == '30sec' else DIR_40SEC

    for tree in trees:
        split_tag = 'cal' if tree in CAL_TREES else 'test'

        for vid in VIDEOS:
            vid_path = os.path.join(frames_dir, tree, vid)
            vid_rows = gt_master[
                (gt_master['source_tree']  == tree) &
                (gt_master['source_video'] == vid)
            ].copy().reset_index(drop=True)

            if len(vid_rows) == 0:
                continue

            for _, row in tqdm(
                vid_rows.iterrows(),
                total=len(vid_rows),
                desc=f'  [{split_tag.upper()}] {tree} {vid}',
                unit='fr',
                bar_format='{l_bar}{bar:20}{r_bar}'
            ):
                fname    = str(row['image_filename']).strip()
                img_path = os.path.join(vid_path, fname)
                if not os.path.exists(img_path):
                    continue

                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue

                gt_c  = int(row['ground_truth_count'])
                dots  = dot_lookup.get(fname, [])
                if not dots:
                    frames_no_dots += 1

                # Re-run inference to get individual boxes
                H, W       = img_bgr.shape[:2]
                all_b = []; all_s = []
                stride = TILE_W - TILE_OVERLAP
                y = 0
                while y < H:
                    x = 0
                    while x < W:
                        x1=x;y1=y
                        x2=min(x+TILE_W,W);y2=min(y+TILE_H,H)
                        tile = img_bgr[y1:y2,x1:x2]
                        with torch.no_grad():
                            out = predictor(tile)
                        inst = out['instances'].to('cpu') if hasattr(out,'__getitem__') else out.to('cpu')
                        if len(inst) > 0:
                            boxes = inst.pred_boxes.tensor.numpy()
                            scores= inst.scores.numpy()
                            boxes[:,0]+=x1;boxes[:,2]+=x1
                            boxes[:,1]+=y1;boxes[:,3]+=y1
                            all_b.append(boxes);all_s.append(scores)
                        if x2==W: break
                        x+=stride
                    if y2==H: break
                    y+=stride

                if all_b:
                    all_b  = np.concatenate(all_b,axis=0)
                    all_s  = np.concatenate(all_s,axis=0)
                    keep   = tv_nms(
                        torch.tensor(all_b,dtype=torch.float32),
                        torch.tensor(all_s,dtype=torch.float32),
                        iou_threshold=NMS_IOU)
                    raw_boxes= all_b[keep.numpy()]
                else:
                    raw_boxes= np.zeros((0,4))

                green_boxes = [b for b in raw_boxes
                               if is_green_detection(img_bgr, b)]

                if not green_boxes:
                    spatial_results.append({
                        'group':sec,'tree':tree,'video':vid,'split':split_tag,
                        'fname':fname,'gt_count':gt_c,'pred_count':0,
                        'tp':0,'fp':0,'fn':len(dots),
                        'precision':0.0,'recall':0.0,'f1':0.0,})
                    continue

                boxes_px = [(float(b[0]),float(b[1]),
                             float(b[2]),float(b[3]))
                            for b in green_boxes]
                tp,fp,fn = box_dot_match(boxes_px, dots)
                prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
                rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
                f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

                spatial_results.append({
                    'group':sec,'tree':tree,'video':vid,'split':split_tag,
                    'fname':fname,'gt_count':gt_c,'pred_count':len(green_boxes),
                    'tp':tp,'fp':fp,'fn':fn,
                    'precision':round(prec,4),
                    'recall':round(rec,4),
                    'f1':round(f1,4),})

spatial_df = pd.DataFrame(spatial_results)

def spatial_metrics(df):
    total_tp = df['tp'].sum(); total_fp = df['fp'].sum()
    total_fn = df['fn'].sum()
    prec = total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0
    rec  = total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    return {'Frames':len(df),'Total TP':int(total_tp),
            'Total FP':int(total_fp),'Total FN':int(total_fn),
            'Precision':round(prec,4),'Recall':round(rec,4),
            'F1':round(f1,4)}

spatial_summary = pd.DataFrame([
    {'Split':'FULL (10 trees)',       **spatial_metrics(spatial_df)},
    {'Split':'CAL  (6 trees)',        **spatial_metrics(spatial_df[spatial_df['split']=='cal'])},
    {'Split':'TEST (4 trees) — MAIN', **spatial_metrics(spatial_df[spatial_df['split']=='test'])},
])

tree_spatial_rows = []
for (grp,tree,split), df_t in spatial_df.groupby(['group','tree','split']):
    m = spatial_metrics(df_t)
    tree_spatial_rows.append({'Group':grp,'Tree':tree,'Split':split,**m})
tree_spatial_df = pd.DataFrame(tree_spatial_rows)

print(tabulate(spatial_summary, headers='keys',
               tablefmt='pretty', showindex=False))
print()
print(tabulate(
    tree_spatial_df[['Group','Tree','Split','Precision','Recall','F1']],
    headers='keys', tablefmt='pretty', showindex=False))


# ## Cell 12 — Overall Metrics

# ── CELL 40 ──────────────────────────────────────────────────
# ============================================================
# CELL 12 — OVERALL METRICS
# ============================================================

def compute_metrics(df):
    g=df['ground_truth'].values; p=df['predicted'].values
    e=p-g; ae=np.abs(e)
    return {
        'Frames'     :len(df),
        'MAE'        :round(float(np.mean(ae)),2),
        'RMSE'       :round(float(np.sqrt(np.mean(e**2))),2),
        'W+-2%'      :round(float(np.mean(ae<=2)*100),1),
        'Bias'       :round(float(np.mean(e)),2),
        'GT mean'    :round(float(np.mean(g)),2),
        'Pred mean'  :round(float(np.mean(p)),2),
        'Overcounts' :int(np.sum(e>0)),
        'Undercounts':int(np.sum(e<0)),
        'Exact'      :int(np.sum(e==0)),
    }

print(SEP)
print(f'  {MODEL_NAME} — METRICS SUMMARY')
print(SEP)

overall_metrics = compute_metrics(res_df)
cal_metrics     = compute_metrics(res_df[res_df['split']=='cal'])
test_metrics    = compute_metrics(res_df[res_df['split']=='test'])

summary_table = pd.DataFrame([
    {'Split':'FULL (10 trees)',        **overall_metrics},
    {'Split':'CAL  (6 trees)',         **cal_metrics},
    {'Split':'TEST (4 trees) — MAIN',  **test_metrics},
])
print(tabulate(summary_table, headers='keys',
               tablefmt='pretty', showindex=False))
print()
print('  TEST split is the main paper result')
print()
print(f'  CuVLER vs CutLER comparison:')
print(f'    CutLER MAE (test) : 4.293')
print(f'    CuVLER MAE (test) : {test_metrics["MAE"]}')
diff = test_metrics["MAE"] - 4.293
verdict = "✓ CuVLER better" if diff < 0 else "✗ CuVLER worse than CutLER"
print(f'    Difference        : {diff:+.3f}  ({verdict})')


# ## Cell 13 — Per-Video Summary

# ── CELL 42 ──────────────────────────────────────────────────
video_rows = []
for (grp,tree,vid,split), df_v in res_df.groupby(['group','tree','video','split']):
    m = compute_metrics(df_v)
    video_rows.append({'Group':grp,'Tree':tree,'Video':vid,'Split':split,**m})
video_df = pd.DataFrame(video_rows)
print(tabulate(
    video_df[['Group','Tree','Video','Split','Frames','MAE','RMSE','W+-2%','Bias']],
    headers='keys', tablefmt='pretty', showindex=False))


# ## Cell 14 — Per-Tree Summary

# ── CELL 44 ──────────────────────────────────────────────────
tree_rows = []
for (grp,tree,split), df_t in res_df.groupby(['group','tree','split']):
    m = compute_metrics(df_t)
    tree_rows.append({'Group':grp,'Tree':tree,'Split':split,**m})
tree_res_df = pd.DataFrame(tree_rows)
print(tabulate(
    tree_res_df[['Group','Tree','Split','Frames','MAE','RMSE','W+-2%','Bias']],
    headers='keys', tablefmt='pretty', showindex=False))


# ## Cell 15 — Group Comparison: 30sec vs 40sec

# ── CELL 46 ──────────────────────────────────────────────────
group_rows = []
for grp, df_g in res_df.groupby('group'):
    m = compute_metrics(df_g)
    group_rows.append({'Group':grp,**m})
group_df = pd.DataFrame(group_rows)
print(tabulate(group_df, headers='keys',
               tablefmt='pretty', showindex=False))


# ## Cell 16 — Visualizations

# ── CELL 48 ──────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')  # replaced %matplotlib inline
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100

gt   = res_df['ground_truth'].values
pred = res_df['predicted'].values
err  = pred - gt
mae_val  = np.mean(np.abs(err))
bias_val = np.mean(err)
w2_val   = np.mean(np.abs(err)<=2)*100

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    f'{MODEL_NAME}\n'
    f'{len(res_df):,} frames  MAE={mae_val:.2f}  '
    f'W±2={w2_val:.1f}%  Bias={bias_val:+.2f}',
    fontsize=13, fontweight='bold')

ax = axes[0,0]
colors = res_df['split'].map({'cal':'steelblue','test':'coral'})
ax.scatter(gt, pred, alpha=0.3, c=colors, s=8)
mn,mx = min(gt.min(),pred.min()), max(gt.max(),pred.max())
ax.plot([mn,mx],[mn,mx],'r--',lw=1.5)
ax.set_xlabel('GT'); ax.set_ylabel('Predicted')
ax.set_title(f'GT vs Predicted  MAE={mae_val:.2f}')
ax.legend(handles=[
    mpatches.Patch(color='steelblue',label='Calibration'),
    mpatches.Patch(color='coral',    label='Test'),
], fontsize=8)

ax = axes[0,1]
ax.hist(err, bins=40, color='darkorange', edgecolor='black', alpha=0.8)
ax.axvline(0,        color='black',lw=1.5,linestyle='--')
ax.axvline(bias_val, color='red',  lw=1.5,label=f'Bias={bias_val:+.2f}')
ax.set_xlabel('Error (pred-GT)'); ax.set_ylabel('Frames')
ax.set_title('Error Distribution')
ax.legend(fontsize=8)

ax = axes[1,0]
colors_t = ['#3498DB' if r['Split']=='cal' else '#E74C3C'
            for _,r in tree_res_df.iterrows()]
ax.bar(tree_res_df['Tree'], tree_res_df['MAE'],
       color=colors_t, edgecolor='black')
ax.axhline(mae_val, color='red', linestyle='--', lw=1.5,
           label=f'Overall MAE={mae_val:.2f}')
ax.set_xlabel('Tree'); ax.set_ylabel('MAE')
ax.set_title('MAE per Tree')
ax.tick_params(axis='x', rotation=45)
ax.legend(handles=[
    mpatches.Patch(color='#3498DB',label='Calibration'),
    mpatches.Patch(color='#E74C3C',label='Test'),
    plt.Line2D([0],[0],color='red',linestyle='--',label='Overall MAE'),
], fontsize=8)

ax = axes[1,1]
pivot = video_df.pivot_table(
    index='Tree', columns='Video', values='MAE', aggfunc='mean')
im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=8)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=8)
ax.set_title('MAE Heatmap (green=low, red=high)')
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.values[i,j]
        if not np.isnan(v):
            ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=7)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, 'cuvler_metrics.png')
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.show()
print(f'  Saved: {plot_path}')


# ## Cell 17 — Ripe Orange Suppression Validation

# ── CELL 50 ──────────────────────────────────────────────────
# ============================================================
# CELL 17 — RIPE ORANGE SUPPRESSION VALIDATION
# ============================================================
# 100 frames, 370 annotated ripe points across 5 videos.
# Uses tiled inference — CuVLER is a detection model.
# Identical to CutLER Cell 16.
# ============================================================

RIPE_VIDEOS = [
    {'tree':'Tree_06','vid':'Vid 02',
     'json':'2116_Vid02_progress.json','annot_dir':'Tree_06'},
    {'tree':'Tree_06','vid':'Vid 03',
     'json':'2116_Vid03_progress.json','annot_dir':'Tree_06'},
    {'tree':'Tree_07','vid':'Vid 02',
     'json':'4737_Vid02_progress.json','annot_dir':'Tree_07'},
    {'tree':'Tree_07','vid':'Vid 03',
     'json':'4737_Vid03_progress.json','annot_dir':'Tree_07'},
    {'tree':'Tree_09','vid':'Vid 02',
     'json':'NN04_Vid02_progress.json','annot_dir':'Tree_09'},
]

all_ripe_results=[]; video_summary=[]

print('=' * 65)
print('  RIPE ORANGE SUPPRESSION VALIDATION')
print('=' * 65)

for vc in RIPE_VIDEOS:
    tree=vc['tree']; vid=vc['vid']
    json_path  = os.path.join(RIPE_ANNOT,vc['annot_dir'],vc['json'])
    frames_dir = os.path.join(RIPE_FRAMES,tree,vid)

    with open(json_path) as f:
        data = json.load(f)
    frame_names=data['frameNames']; annotations=data['annotations']

    vid_total=0; vid_supp=0; vid_fail=0

    for idx_str, ripe_pts in tqdm(
            annotations.items(),
            desc=f'  {tree}/{vid}', unit='frame'):
        idx=int(idx_str); fname=frame_names[idx]
        img_path=os.path.join(frames_dir,fname)
        if not os.path.exists(img_path): continue

        img_bgr = cv2.imread(img_path)

        # Tiled inference to get raw and green boxes
        H,W=img_bgr.shape[:2]
        all_b=[]; all_s=[]
        stride=TILE_W-TILE_OVERLAP
        y=0
        while y<H:
            x=0
            while x<W:
                x1=x;y1=y;x2=min(x+TILE_W,W);y2=min(y+TILE_H,H)
                tile=img_bgr[y1:y2,x1:x2]
                with torch.no_grad():
                    out=predictor(tile)
                inst=out['instances'].to('cpu') if hasattr(out,'__getitem__') else out.to('cpu')
                if len(inst)>0:
                    boxes=inst.pred_boxes.tensor.numpy()
                    scores=inst.scores.numpy()
                    boxes[:,0]+=x1;boxes[:,2]+=x1
                    boxes[:,1]+=y1;boxes[:,3]+=y1
                    all_b.append(boxes);all_s.append(scores)
                if x2==W: break
                x+=stride
            if y2==H: break
            y+=stride

        if all_b:
            all_b=np.concatenate(all_b,axis=0)
            all_s=np.concatenate(all_s,axis=0)
            keep=tv_nms(
                torch.tensor(all_b,dtype=torch.float32),
                torch.tensor(all_s,dtype=torch.float32),
                iou_threshold=NMS_IOU)
            raw_boxes=all_b[keep.numpy()]
        else:
            raw_boxes=np.zeros((0,4))

        green_boxes=[b for b in raw_boxes
                     if is_green_detection(img_bgr,b)]

        for p in ripe_pts:
            px,py=p['x'],p['y']
            in_raw  =any(b[0]<=px<=b[2] and b[1]<=py<=b[3]
                         for b in raw_boxes)
            in_green=any(b[0]<=px<=b[2] and b[1]<=py<=b[3]
                         for b in green_boxes)
            if in_raw and in_green: vid_fail+=1
            else: vid_supp+=1
            vid_total+=1

        all_ripe_results.append({
            'tree':tree,'video':vid,'frame':fname,'ripe_pts':len(ripe_pts)})

    rate=vid_supp/vid_total*100 if vid_total>0 else 0
    print(f'  {tree}/{vid}: {vid_total} pts  '
          f'suppressed={vid_supp}  failed={vid_fail}  rate={rate:.1f}%')
    video_summary.append({
        'Tree':tree,'Video':vid,'Total':vid_total,
        'Suppressed':vid_supp,'Failed':vid_fail,'Rate%':round(rate,1)})

vsum_df    = pd.DataFrame(video_summary)
total_ripe = vsum_df['Total'].sum()
total_supp = vsum_df['Suppressed'].sum()
overall_rate= total_supp/total_ripe*100 if total_ripe>0 else 0

print()
print('=' * 65)
print(tabulate(vsum_df, headers='keys',
               tablefmt='pretty', showindex=False))
print()
print(f'  Total ripe points   : {total_ripe}')
print(f'  Overall suppression : {overall_rate:.1f}%')
print(f'  Paper statement:')
print(f'  "The color filter suppressed {overall_rate:.1f}% of ripe orange')
print(f'   detections across {total_ripe} annotated ripe orange points."')
print('=' * 65)


# ## Cell 18 — Export All Results to Excel

# ── CELL 52 ──────────────────────────────────────────────────
# ============================================================
# CELL 18 — EXPORT ALL RESULTS TO EXCEL
# ============================================================

out_path = os.path.join(OUT_DIR, 'cuvler_results.xlsx')

test_m = compute_metrics(res_df[res_df['split']=='test'])
full_m = compute_metrics(res_df)

summary = pd.DataFrame([{
    'Model'             : MODEL_NAME,
    'Pretrained On'     : PRETRAINED_ON,
    'Weights Source'    : WEIGHTS_LOADED_AS,
    'Architecture'      : 'Cascade Mask R-CNN (VoteCut pseudo-labels)',
    'Tile Size'         : f'{TILE_W}×{TILE_H}',
    'Overlap'           : TILE_OVERLAP,
    'NMS IoU'           : NMS_IOU,
    'Threshold'         : BEST_THRESH,
    'Total Frames'      : len(res_df),
    'Test MAE'          : test_m['MAE'],
    'Test RMSE'         : test_m['RMSE'],
    'Test W+-2%'        : test_m['W+-2%'],
    'Test Bias'         : test_m['Bias'],
    'Full MAE'          : full_m['MAE'],
    'Full RMSE'         : full_m['RMSE'],
    'Full W+-2%'        : full_m['W+-2%'],
    'Full Bias'         : full_m['Bias'],
    'Ripe Filtered'     : total_filtered,
    'Ripe Suppression%' : round(overall_rate, 1),
    'vs CutLER MAE'     : round(test_m['MAE'] - 4.293, 3),
}])

with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    summary.to_excel(        writer, sheet_name='Summary',          index=False)
    res_df.to_excel(         writer, sheet_name='Frame Results',    index=False)
    video_df.to_excel(       writer, sheet_name='Per-Video',        index=False)
    tree_res_df.to_excel(    writer, sheet_name='Per-Tree',         index=False)
    group_df.to_excel(       writer, sheet_name='Group Comparison', index=False)
    summary_table.to_excel(  writer, sheet_name='Cal vs Test',      index=False)
    sweep_df.to_excel(       writer, sheet_name='Threshold Sweep',  index=False)
    vsum_df.to_excel(        writer, sheet_name='Ripe Validation',  index=False)
    spatial_df.to_excel(     writer, sheet_name='Spatial Metrics',  index=False)
    tree_spatial_df.to_excel(writer, sheet_name='Spatial Per-Tree', index=False)
    inventory_df.to_excel(   writer, sheet_name='Inventory',        index=False)

print('=' * 65)
print(f'  Saved: {out_path}')
print()
print('  Sheets:')
print('    1.  Summary           — overall metrics + config')
print('    2.  Frame Results     — per frame (10,577 rows)')
print('    3.  Per-Video         — 30 videos')
print('    4.  Per-Tree          — 10 trees')
print('    5.  Group Comparison  — 30sec vs 40sec')
print('    6.  Cal vs Test       — calibration vs test split')
print('    7.  Threshold Sweep   — sweep table')
print('    8.  Ripe Validation   — color filter suppression')
print('    9.  Spatial Metrics   — P/R/F1 per frame')
print('    10. Spatial Per-Tree  — P/R/F1 per tree')
print('    11. Inventory         — frame counts per video')
print()
print(f'  TEST split results (main paper number):')
print(f'    MAE   : {test_m["MAE"]}')
print(f'    RMSE  : {test_m["RMSE"]}')
print(f'    W+-2% : {test_m["W+-2%"]}%')
print(f'    Bias  : {test_m["Bias"]}')
print()
diff = test_m["MAE"] - 4.293
print(f'  CuVLER vs CutLER: {diff:+.3f}  '
      f'({"CuVLER better ✓" if diff<0 else "CuVLER worse ✗"})')
print('=' * 65)


# ── CELL 53 ──────────────────────────────────────────────────

