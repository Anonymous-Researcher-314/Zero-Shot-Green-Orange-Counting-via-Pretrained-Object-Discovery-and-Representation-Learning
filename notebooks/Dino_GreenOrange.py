# 04_dino.py
# Converted from 04_dino.ipynb

# # 04 - DINO
#
# **Run `00_shared.ipynb` first.**
#
# **Model:** DINO (Self-DIstillation with NO labels) · ICCV 2021 · Meta AI
# **Type:** Self-Supervised Vision Transformer - Attention-based detection
# **Backbone:** ViT-S/8 - same weights used in the original CSUN study
#
# ---
# ### What DINO does
# DINO trains a Vision Transformer using **self-distillation** - a student network learns to match a momentum-updated teacher with no labels at all. The [CLS] token attention maps have a strong emergent property: they highlight semantically meaningful regions.
#
# For orange detection we extract multi-head attention maps, average them, threshold the result, and run **blob detection** to count individual fruit.
#
# ### Pipeline
# 1. Resize 4K frame to 480x480
# 2. Run DINO ViT-S/8 forward pass - CLS attention from all 6 heads
# 3. Average across heads - normalize to [0, 1]
# 4. Threshold at `thr * max(attn)` - morphological close + open (7x7 ellipse)
# 5. Connected components - area filter [min_area, 12000] - count blobs
#
# ---

# ---
# ## Cell 1 - Load Shared Data
# > Loads ground truth counts, dot annotations, 90-frame sweep set, and video groups from `00_shared.ipynb`.

# -- Load shared data ----------------------------------------------------------
import sys
sys.path.insert(0, r'C:\orange_project\notebooks')
from shared import load_cache, mae, rmse, bias, within_n

data         = load_cache()
gt_lookup    = data['gt_lookup']
dot_lookup   = data['dot_lookup']
sweep_frames = data['sweep_frames']
video_groups = data['video_groups']

print(f'Sweep frames : {len(sweep_frames)}')
print(f'GT lookup    : {len(gt_lookup):,} frames')
print(f'Dot lookup   : {len(dot_lookup):,} annotated frames')

# -- Tree ID renaming map -------------------------------------------------------
# Maps internal video/tree IDs to display labels (Tree 01 ... Tree 10)
TREE_ID_MAP = {
    'Tree_483'  : 'Tree 01',
    'Tree_484'  : 'Tree 02',
    'Tree_490'  : 'Tree 03',
    'Tree_NN01' : 'Tree 04',
    'Tree_NN02' : 'Tree 05',
    'Tree_2216' : 'Tree 06',
    'Tree_4737' : 'Tree 07',
    'Tree_NN03' : 'Tree 08',
    'Tree_NN04' : 'Tree 09',
    'Tree_NN05' : 'Tree 10',
}

def tree_label(vid_id):
    """Return display label for a video/tree ID, or the original if not in map."""
    return TREE_ID_MAP.get(vid_id, vid_id)

# ---
# ## Cell 2 - Libraries and Load DINO Model
# > Installs dependencies, loads DINO ViT-S/8 from the torch.hub cache, and builds a filename-to-path index for all 10,577 frames.

import subprocess, sys, os, glob, time
subprocess.check_call([sys.executable,'-m','pip','install','scikit-image','scipy'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import numpy as np
import cv2
import torch
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# DINO ViT-S/8 -- same model as 75-frame study
print('Loading DINO vit_small patch8 ...')
model_dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', pretrained=True)
model_dino.eval().to(DEVICE)
N_HEADS = model_dino.blocks[-1].attn.num_heads
print(f'DINO ViT-S/8 loaded  |  heads={N_HEADS}')

# frame index
FRAMES_ROOT = r'C:\orange_project\frames'
frame_path = {}
for p in glob.glob(os.path.join(FRAMES_ROOT,'**','*.jpg'),recursive=True):
    frame_path[os.path.basename(p)] = p
print(f'Indexed {len(frame_path):,} frames')

# ---
# ## Stage 1 - Parameter Sweep on 90 Frames
#
# Runs the full DINO attention pipeline on 90 sweep frames across **30 parameter combinations**
# (5 thresholds x 6 min_area values). Defines `d1_attention()`, `d1_blobs()`, `d1_dot_f1()`.
#
# Progress is printed every 10 frames with a ranked summary table (MAE, Bias, W+-2, F1).
# Results saved to `cache/dino_s1_90.pkl`.

# -- Stage 1 -- Exact 75-frame study replication on 90 sweep frames ------------
#
# Exactly what Notebook_02_DINO did:
#   resize full frame to 480x480 (IMG_SIZE=480, patch=8 -> 60x60=3600 patches)
#   get_last_selfattention() -> CLS row -> average 6 heads -> normalize [0,1]
#   threshold at frac * max(attn_map)
#   morphological close + open (7x7 ellipse kernel)
#   connected components -> area filter [min_area, 12000]
#   work in 480x480 space, scale dot coords for F1
#
# Sweep: ATTN_THRESH x MIN_AREA on all 90 frames

import pickle, itertools
from scipy import ndimage as ndi

D1_SIZE    = 480
D1_PATCH   = 8
D1_GRID    = D1_SIZE // D1_PATCH   # 60
D1_MAX_AREA = 12000

D1_MEAN = torch.tensor([0.485,0.456,0.406],device=DEVICE).view(1,3,1,1)
D1_STD  = torch.tensor([0.229,0.224,0.225],device=DEVICE).view(1,3,1,1)

# -- extract attention map at 480x480 (exact study method)
def d1_attention(img_bgr):
    resized = cv2.resize(img_bgr, (D1_SIZE, D1_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    t   = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(DEVICE)
    t   = (t - D1_MEAN) / D1_STD
    with torch.no_grad():
        attn = model_dino.get_last_selfattention(t)  # (1, 6, 3601, 3601)
    attn     = attn[0]                # (6, 3601, 3601)
    cls_attn = attn[:, 0, 1:]         # (6, 3600) -- CLS row, patch columns
    cls_attn = cls_attn.reshape(N_HEADS, D1_GRID, D1_GRID)  # (6, 60, 60)
    avg      = cls_attn.mean(0).cpu().numpy()                # (60, 60)
    avg      = (avg - avg.min()) / (avg.max() - avg.min() + 1e-8)  # normalize [0,1]
    # upsample to 480x480 (same as study upsampling to original -- our 'original' is 480x480)
    return cv2.resize(avg, (D1_SIZE, D1_SIZE), interpolation=cv2.INTER_LINEAR)

# -- blob detection (exact study)
def d1_blobs(attn_map, thresh_frac, min_area, max_area=D1_MAX_AREA):
    thresh_val = thresh_frac * attn_map.max()
    binary     = (attn_map >= thresh_val).astype(np.uint8) * 255
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    binary     = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary     = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blobs = []
    for i in range(1, n_lbl):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            x=stats[i,cv2.CC_STAT_LEFT]; y=stats[i,cv2.CC_STAT_TOP]
            w=stats[i,cv2.CC_STAT_WIDTH]; h=stats[i,cv2.CC_STAT_HEIGHT]
            blobs.append({'bbox':[x,y,x+w,y+h],'area':area})
    return blobs

# -- F1: scale dot coords from 4K to 480x480
def d1_dot_f1(blobs, dots, orig_w, orig_h):
    if not dots and not blobs: return 1.0,1.0,1.0
    if not blobs: return 0.0,0.0,0.0
    if not dots:  return 0.0,0.0,0.0
    sx=D1_SIZE/orig_w; sy=D1_SIZE/orig_h
    md,mb=set(),set()
    for bi,blob in enumerate(blobs):
        x1,y1,x2,y2=blob['bbox']
        for di,dot in enumerate(dots):
            dx=float(dot['x'])*sx; dy=float(dot['y'])*sy
            if x1<=dx<=x2 and y1<=dy<=y2: md.add(di); mb.add(bi)
    tp=len(md); fp=len(blobs)-len(mb); fn=len(dots)-len(md)
    pre=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
    return pre,rec,2*pre*rec/(pre+rec+1e-9)

# -- configs (exact study sweep grid)
D1_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
D1_MIN_AREAS  = [100, 200, 400, 800, 1500, 3000]
configs_d1    = list(itertools.product(D1_THRESHOLDS, D1_MIN_AREAS))  # 30 configs
print(f'{len(configs_d1)} configs: {len(D1_THRESHOLDS)} thresholds x {len(D1_MIN_AREAS)} min_areas')
print(f'Input: 480x480  |  max_area={D1_MAX_AREA}  |  morph kernel 7x7 ellipse\n')

# -- summary
def print_d1(label, accum, gts):
    g=np.array(gts); rows=[]
    for cfg,s in accum.items():
        p=np.array(s['preds'])
        rows.append((np.mean(np.abs(p-g)),cfg,np.mean(p-g),
                     np.mean(np.abs(p-g)<=2)*100,
                     np.mean(s['prec']),np.mean(s['rec']),np.mean(s['f1'])))
    rows.sort()
    print('\n'+'='*72)
    print(f'  {label}   n={len(g)}   GT mean={g.mean():.1f}   (top 20 by MAE)')
    print(f'  {"thr":>5} {"mina":>5} | {"MAE":>6} {"Bias":>6} {"W+-2%":>6} | {"Prec":>5} {"Rec":>5} {"F1":>5}')
    print('  '+'-'*68)
    for (mae_v,cfg,bia,w2,pre,rec,f1) in rows[:20]:
        t,m=cfg
        print(f'  {t:5.2f} {m:5d} | {mae_v:6.2f} {bia:+6.2f} {w2:6.1f}% | {pre:5.3f} {rec:5.3f} {f1:5.3f}')
    print('='*72+'\n')

# -- main loop
print(f'Running {len(sweep_frames)} frames -- evaluation every 10 frames\n')
accum_d1={cfg:{'preds':[],'prec':[],'rec':[],'f1':[]} for cfg in configs_d1}
gts_d1=[]; t_start=time.time()

for i,fname in enumerate(sweep_frames):
    img=cv2.imread(frame_path[fname])
    H,W=img.shape[:2]
    gt=gt_lookup[fname]
    dots=dot_lookup.get(fname,[])
    gts_d1.append(gt)

    t0=time.time()
    attn=d1_attention(img)   # 480x480 normalized attention map
    t_gpu=time.time()-t0

    t1=time.time()
    for (thr,min_a) in configs_d1:
        blobs=d1_blobs(attn,thr,min_a)
        pre,rec,f1=d1_dot_f1(blobs,dots,W,H)
        accum_d1[(thr,min_a)]['preds'].append(len(blobs))
        accum_d1[(thr,min_a)]['prec'].append(pre)
        accum_d1[(thr,min_a)]['rec'].append(rec)
        accum_d1[(thr,min_a)]['f1'].append(f1)
    t_cpu=time.time()-t1

    elapsed=time.time()-t_start; eta=elapsed/(i+1)*(len(sweep_frames)-i-1)
    print(f'  [{i+1:2d}/90] {fname:<42}  GT={gt:3d}  gpu={t_gpu:.1f}s cpu={t_cpu:.2f}s  ETA {eta/60:.0f}m')
    if (i+1)%10==0: print_d1(f'After frame {i+1}/90',accum_d1,gts_d1)

print_d1('FINAL -- all 90 frames',accum_d1,gts_d1)
with open(r'C:\orange_project\notebooks\cache\dino_s1_90.pkl','wb') as f:
    pickle.dump({'accum':accum_d1,'gts':gts_d1,'sweep_frames':sweep_frames,'configs':configs_d1},f)
print('Saved -> cache/dino_s1_90.pkl')

# ---
# ## Stage 2 - Full Run on All 10,577 Frames
#
# Loads the best config from Stage 1 and runs DINO on every frame.
# If the full results are already saved, loads instantly - no re-run needed.
# If interrupted, resumes from the last checkpoint automatically.
#
# Progress: every 10 frames for the first 100, then every 1000 frames with elapsed time and ETA.
# Saves to `cache/dino_s1_full.pkl`.

# -- Stage 1 Full Run - All 10,577 Frames -------------------------------------
#
#  Uses the functions already defined above:  d1_attention()  d1_blobs()  d1_dot_f1()
#
#  What this cell does:
#    1. Loads the best (thr, min_area) config found on the 90-frame sweep
#    2. Runs DINO Stage 1 on every frame in frame_path  (one GPU call per frame)
#    3. For annotated frames (those in dot_lookup) it also computes Prec / Rec / F1
#       - done inside the main loop so nothing needs to be re-run afterwards
#    4. Prints progress every 50 frames:  [done/total]  %  |  s/frame  |  ETA  |  finish HH:MM
#    5. Saves a checkpoint every 50 frames  ->  safe to stop and resume
#    6. Prints a final summary and saves  dino_s1_full.pkl
#
#  Estimated time on RTX 3080:  ~76 ms/frame  ->  ~13 minutes for 10,577 frames

import pickle, time, os
import numpy as np
from datetime import datetime, timedelta

S1_FULL_PKL      = r'C:\orange_project\notebooks\cache\dino_s1_full.pkl'
CHECKPOINT_EVERY = 50

# -- 1. Load best config from the 90-frame sweep ------------------------------
with open(r'C:\orange_project\notebooks\cache\dino_s1_90.pkl', 'rb') as f:
    s1_tr = pickle.load(f)

# Pick config with the highest mean F1 on annotated sweep frames
annot_idx = [i for i, fn in enumerate(s1_tr['sweep_frames']) if fn in dot_lookup]
BEST_CFG  = max(
    s1_tr['accum'].keys(),
    key=lambda c: np.mean([s1_tr['accum'][c]['f1'][i] for i in annot_idx])
)
BEST_THR, BEST_MIN_A = BEST_CFG
TRAIN_F1 = np.mean([s1_tr['accum'][BEST_CFG]['f1'][i] for i in annot_idx])
print(f'Best config  :  thr={BEST_THR}   min_area={BEST_MIN_A}')
print(f'Train F1     :  {TRAIN_F1:.3f}  (on {len(annot_idx)} annotated sweep frames)')

# -- 2. Frame lists ------------------------------------------------------------
all_frames = list(frame_path.keys())
annotated  = set(fn for fn in all_frames if fn in dot_lookup)
print(f'\nTotal frames : {len(all_frames):,}')
print(f'Annotated    : {len(annotated):,}')

# -- 3. Resume support ---------------------------------------------------------
if os.path.exists(S1_FULL_PKL):
    with open(S1_FULL_PKL, 'rb') as f:
        _cache = pickle.load(f)
    preds_all  = _cache['preds_all']        # fname -> predicted count
    f1_store   = _cache.get('f1_store', {}) # fname -> (prec, rec, f1)
    print(f'\nCheckpoint found - {len(preds_all):,}/{len(all_frames):,} done. Resuming...')
else:
    preds_all = {}
    f1_store  = {}
    print('\nNo checkpoint found - starting from scratch.')

remaining    = [fn for fn in all_frames if fn not in preds_all]
already_done = len(all_frames) - len(remaining)

est_s = len(remaining) * 0.076
print(f'\nFrames left  : {len(remaining):,}')
print(f'Est. time    : ~{est_s/60:.0f} min  ({est_s:.0f} s)')
print(f'Checkpoint   : every {CHECKPOINT_EVERY} frames\n')
print('-' * 72)

# -- 4. Main loop --------------------------------------------------------------
t0 = time.time()

for idx, fname in enumerate(remaining):

    img = cv2.imread(frame_path[fname])
    if img is None:
        preds_all[fname] = 0
        if fname in annotated:
            f1_store[fname] = (0.0, 0.0, 0.0)
    else:
        H, W = img.shape[:2]
        attn  = d1_attention(img)
        blobs = d1_blobs(attn, BEST_THR, BEST_MIN_A)
        preds_all[fname] = len(blobs)

        if fname in annotated:
            pre, rec, f1 = d1_dot_f1(blobs, dot_lookup[fname], W, H)
            f1_store[fname] = (pre, rec, f1)

    # -- progress + checkpoint every 50 frames --------------------------------
    if (idx + 1) % CHECKPOINT_EVERY == 0 or (idx + 1) == len(remaining):

        el         = time.time() - t0
        done_now   = idx + 1
        rate       = el / done_now                          # s per frame
        eta_s      = rate * (len(remaining) - done_now)
        finish_dt  = datetime.now() + timedelta(seconds=eta_s)
        total_done = already_done + done_now
        pct        = total_done / len(all_frames) * 100

        print(f'  [{total_done:5d}/{len(all_frames)}]  {pct:5.1f}%  |  '
              f'{rate:.2f} s/frame  |  '
              f'elapsed {el/60:.1f} m  |  '
              f'ETA {eta_s/60:.0f} m  |  '
              f'finish ~{finish_dt.strftime("%H:%M")}')

        with open(S1_FULL_PKL, 'wb') as f:
            pickle.dump({
                'preds_all' : preds_all,
                'f1_store'  : f1_store,
                'thr'       : BEST_THR,
                'min_area'  : BEST_MIN_A,
                'completed' : total_done,
            }, f)

# -- 5. Final summary ----------------------------------------------------------
ann_fns = [fn for fn in all_frames if fn in annotated]
p_ann   = np.array([preds_all[fn] for fn in ann_fns])
g_ann   = np.array([gt_lookup[fn] for fn in ann_fns])

mae_v  = np.mean(np.abs(p_ann - g_ann))
bias_v = np.mean(p_ann - g_ann)
w2     = np.mean(np.abs(p_ann - g_ann) <= 2) * 100

prs = [f1_store[fn][0] for fn in ann_fns if fn in f1_store]
rcs = [f1_store[fn][1] for fn in ann_fns if fn in f1_store]
f1s = [f1_store[fn][2] for fn in ann_fns if fn in f1_store]

total_elapsed = time.time() - t0

print('\n' + '=' * 60)
print(f'  DINO Stage 1 - Full {len(all_frames):,} Frames')
print(f'  Config  :  thr={BEST_THR}   min_area={BEST_MIN_A}')
print(f'  Eval on :  {len(ann_fns):,} annotated frames')
print('-' * 60)
print(f'  MAE    = {mae_v:.2f}')
print(f'  Bias   = {bias_v:+.2f}')
print(f'  W +- 2  = {w2:.1f} %')
print(f'  Prec   = {np.mean(prs):.3f}')
print(f'  Rec    = {np.mean(rcs):.3f}')
print(f'  F1     = {np.mean(f1s):.3f}')
print('-' * 60)
print(f'  Avg pred (all frames) = {np.mean(list(preds_all.values())):.1f} oranges / frame')
print(f'  Total time            = {total_elapsed/60:.1f} min')
print('=' * 60)

# -- 6. Save final result ------------------------------------------------------
with open(S1_FULL_PKL, 'wb') as f:
    pickle.dump({
        'preds_all' : preds_all,
        'f1_store'  : f1_store,
        'thr'       : BEST_THR,
        'min_area'  : BEST_MIN_A,
        'completed' : len(all_frames),
        'annotated' : ann_fns,
        'mae'       : mae_v,
        'bias'      : bias_v,
        'w2'        : w2,
        'prec'      : np.mean(prs),
        'rec'       : np.mean(rcs),
        'f1'        : np.mean(f1s),
    }, f)
print(f'\n  Saved  ->  {S1_FULL_PKL}')

# ---
# ## Stage 2b - Per-Video Breakdown
#
# Prints MAE, Bias, W±2%, RMSE, and F1 for each of the 30 videos — sorted by MAE (best first).
# Uses the same `preds_all` and `f1_store` computed in Stage 2.

# -- Per-Video Breakdown -------------------------------------------------------
import numpy as np

print()
print('=' * 75)
print('  DINO - Per-Video Breakdown  (all 10,577 frames)')
print(f'  Config : thr={BEST_THR}   min_area={BEST_MIN_A}')
print('=' * 75)
print(f'  {"Video":<24}  {"n":>5}  {"GT avg":>7}  {"MAE":>7}  {"Bias":>7}  {"W+-2%":>6}  {"RMSE":>7}  {"F1":>6}')
print('  ' + '-' * 71)

_vid_rows = []
for _vid_id, _fnames in sorted(video_groups.items()):
    _fns = [f for f in _fnames if f in preds_all and f in gt_lookup]
    if not _fns:
        continue
    _ps  = np.array([preds_all[f] for f in _fns])
    _gs  = np.array([gt_lookup[f] for f in _fns])
    _f1v = [f1_store[f][2] for f in _fns if f in f1_store]
    _vid_rows.append({
        'vid':    tree_label(_vid_id),
        'n':      len(_fns),
        'gt_avg': float(np.mean(_gs)),
        'mae':    float(np.mean(np.abs(_ps - _gs))),
        'bias':   float(np.mean(_ps - _gs)),
        'w2':     float(np.mean(np.abs(_ps - _gs) <= 2) * 100),
        'rmse':   float(np.sqrt(np.mean((_ps - _gs) ** 2))),
        'f1':     float(np.mean(_f1v)) if _f1v else 0.0,
    })

_vid_rows.sort(key=lambda r: r['mae'])
for r in _vid_rows:
    print(f'  {r["vid"]:<24}  {r["n"]:>5}  {r["gt_avg"]:>7.1f}  '
          f'{r["mae"]:>7.2f}  {r["bias"]:>+7.2f}  {r["w2"]:>5.1f}%  '
          f'{r["rmse"]:>7.2f}  {r["f1"]:>6.3f}')
print('=' * 75)

# ---
# ## Stage 3 - Analysis: Edge Hypothesis and Video Continuity
#
# **Edge hypothesis:** Are oranges near the frame border harder to detect?
# Groups frames into 4 quartiles by the fraction of their GT dots that fall in the edge zone
# (within 15% of any 4K border). Compares mean F1 and MAE across quartiles.
#
# **Video continuity:** Are errors isolated (one bad frame) or persistent (runs of bad frames)?
# Sorts frames within each video by frame number, classifies each miss as isolated or persistent.

# -- Analysis: Edge Hypothesis + Video Continuity  (DINO Stage 1) -------------
#
#  ANALYSIS 1 - Edge Orange Hypothesis
#    Question : Oranges near the frame border - are they harder to detect?
#    Method   : For every annotated frame, count what fraction of its GT dots
#               fall inside the "edge zone" (within 15% of any 4K border).
#               Group frames into 4 quartiles by that fraction.
#               Compare mean F1 and MAE across quartiles.
#               A clear F1 drop from Q1 -> Q4 = edge oranges are harder.
#
#  ANALYSIS 2 - Video Continuity
#    Question : If a frame has a big error, do the frames just before/after also?
#    Method   : For each video, sort frames by frame number.
#               Label each frame "miss" if |pred - gt| > MISS_THRESH.
#               Classify each miss as:
#                 isolated   - frame misses, both neighbours (F-1, F+1) are OK
#                 persistent - at least one neighbour also misses
#               High persistent% = errors come in stretches, not random noise.

import pickle, re, os
import numpy as np
import matplotlib.pyplot as plt

MODEL = 'DINO Stage 1'

# -- Load DINO full results ----------------------------------------------------
with open(r'C:\orange_project\notebooks\cache\dino_s1_full.pkl', 'rb') as f:
    _full = pickle.load(f)
preds_all = _full['preds_all']   # fname -> int
f1_store  = _full['f1_store']    # fname -> (prec, rec, f1)

# ==============================================================================
#  ANALYSIS 1 - Edge Orange Hypothesis
# ==============================================================================
IMG_W, IMG_H = 3840, 2160
EDGE_FRAC    = 0.15                   # within 15% of any border = edge zone
EX           = IMG_W * EDGE_FRAC      # 576 px left/right
EY           = IMG_H * EDGE_FRAC      # 324 px top/bottom

def dot_is_edge(d):
    x, y = float(d['x']), float(d['y'])
    return x < EX or x > IMG_W - EX or y < EY or y > IMG_H - EY

records = []
for fname, dots in dot_lookup.items():
    if fname not in f1_store or fname not in preds_all or len(dots) == 0:
        continue
    n_edge   = sum(1 for d in dots if dot_is_edge(d))
    ef       = n_edge / len(dots)
    _, _, f1 = f1_store[fname]
    gt       = gt_lookup.get(fname, 0)
    pred     = preds_all[fname]
    records.append((ef, f1, abs(pred - gt), pred - gt))

records.sort(key=lambda r: r[0])
n  = len(records)
qs = [records[i*n//4 : (i+1)*n//4] for i in range(4)]
qlabels = ['Q1 Center (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 Edge (75-100%)']

print(f'{"="*68}')
print(f'  {MODEL}  -  ANALYSIS 1: Edge Orange Hypothesis')
print(f'  Edge zone : within {EDGE_FRAC*100:.0f}% of any border')
print(f'              x < {EX:.0f}  or  x > {IMG_W-EX:.0f}   |   y < {EY:.0f}  or  y > {IMG_H-EY:.0f}')
print(f'  Frames    : {n:,}')
print(f'{"="*68}')
print(f'  {"Group":<22}  {"n":>5}  {"avg edge%":>10}  {"mean F1":>8}  {"MAE":>7}  {"Bias":>7}')
print('  ' + '-'*62)
f1_per_q = []
for grp, lbl in zip(qs, qlabels):
    ef_m   = np.mean([r[0] for r in grp]) * 100
    f1_m   = np.mean([r[1] for r in grp])
    mae_m  = np.mean([r[2] for r in grp])
    bias_m = np.mean([r[3] for r in grp])
    f1_per_q.append(f1_m)
    print(f'  {lbl:<22}  {len(grp):>5}  {ef_m:>9.1f}%  {f1_m:>8.3f}  {mae_m:>7.2f}  {bias_m:>+7.2f}')

f1_drop = f1_per_q[0] - f1_per_q[-1]
print(f'\n  F1 Q1->Q4 : {" -> ".join(f"{v:.3f}" for v in f1_per_q)}')
if f1_drop > 0.02:
    print(f'  [OK] Edge oranges ARE harder to detect  (F1 drops {f1_drop:.3f})')
elif f1_drop < -0.02:
    print(f'  [NO] Edge oranges are EASIER to detect  (F1 rises {abs(f1_drop):.3f})')
else:
    print(f'  -> No significant edge effect  (difference = {f1_drop:.4f})')
print(f'{"="*68}')

# ==============================================================================
#  ANALYSIS 2 - Video Continuity
# ==============================================================================
MISS_THRESH = 5   # |pred - gt| > 5 = this frame is a "miss"

def frame_num(fname):
    m = re.search(r'_F(\d+)\.jpg$', fname, re.IGNORECASE)
    return int(m.group(1)) if m else 0

isolated = persistent = total_frames = 0
video_rows = []

for vid_id, fnames in video_groups.items():
    fns = [f for f in fnames if f in gt_lookup and f in preds_all]
    if len(fns) < 3:
        continue
    fns.sort(key=frame_num)
    errs    = [abs(preds_all[f] - gt_lookup[f]) for f in fns]
    is_miss = [e > MISS_THRESH for e in errs]
    n_iso = n_pers = 0
    for i in range(1, len(is_miss) - 1):
        if is_miss[i]:
            if not is_miss[i-1] and not is_miss[i+1]:
                n_iso  += 1   # neighbours are fine -> isolated
            else:
                n_pers += 1   # at least one neighbour also misses -> persistent
    isolated   += n_iso
    persistent += n_pers
    total_frames += len(fns)
    video_rows.append({'vid': tree_label(vid_id), 'n': len(fns),
                       'n_miss': sum(is_miss), 'miss_pct': sum(is_miss)/len(fns)*100,
                       'n_iso': n_iso, 'n_pers': n_pers,
                       'mean_err': np.mean(errs)})

video_rows.sort(key=lambda x: -x['miss_pct'])
total_misses = isolated + persistent

print(f'\n{"="*68}')
print(f'  {MODEL}  -  ANALYSIS 2: Video Continuity')
print(f'  Miss threshold : |pred - gt| > {MISS_THRESH}')
print(f'  Videos         : {len(video_rows)}   Frames : {total_frames:,}')
print(f'{"="*68}')
print(f'  Total misses      : {total_misses:,}')
if total_misses > 0:
    print(f'  Isolated misses   : {isolated:,}  ({isolated/total_misses*100:.1f}%)  <- neighbours OK')
    print(f'  Persistent misses : {persistent:,}  ({persistent/total_misses*100:.1f}%)  <- >=1 neighbour also misses')
    pct_pers = persistent / total_misses * 100
    print()
    if pct_pers > 60:
        print(f'  [OK] Mostly PERSISTENT ({pct_pers:.1f}%) - bad angle/light affects stretches of frames.')
    elif pct_pers < 40:
        print(f'  -> Mostly ISOLATED ({100-pct_pers:.1f}%) - each frame behaves independently.')
    else:
        print(f'  ~ Mixed: {pct_pers:.1f}% persistent  /  {100-pct_pers:.1f}% isolated')

print(f'\n  Per-video results (all {len(video_rows)} videos, sorted by miss rate):')
print(f'  {"Video":<22}  {"frames":>7}  {"misses":>7}  {"miss%":>6}  {"iso":>5}  {"pers":>6}  {"mean err":>9}')
print('  ' + '-'*66)
for r in video_rows:
    print(f'  {r["vid"]:<22}  {r["n"]:>7}  {r["n_miss"]:>7}  {r["miss_pct"]:>5.1f}%'
          f'  {r["n_iso"]:>5}  {r["n_pers"]:>6}  {r["mean_err"]:>9.2f}')
print(f'{"="*68}')

# -- Plot ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'{MODEL} - Edge Hypothesis & Video Continuity', fontsize=13, fontweight='bold')

ax = axes[0]
ax.bar(range(4), f1_per_q, color=['#2196F3','#64B5F6','#FF9800','#F44336'],
       edgecolor='white', linewidth=1.2)
ax.set_xticks(range(4))
ax.set_xticklabels(['Q1\nCenter\n(0-25%)', 'Q2\n(25-50%)', 'Q3\n(50-75%)', 'Q4\nEdge\n(75-100%)'],
                   fontsize=9)
ax.set_ylabel('Mean F1', fontsize=11)
ax.set_title('F1 vs Edge Fraction of GT Dots', fontsize=11)
ax.set_ylim(0, max(f1_per_q) * 1.3)
for i, v in enumerate(f1_per_q):
    ax.text(i, v + max(f1_per_q)*0.03, f'{v:.3f}', ha='center', fontsize=10)

ax2 = axes[1]
ax2.bar(['Isolated\n(neighbours OK)', 'Persistent\n(neighbour also misses)'],
        [isolated, persistent], color=['#4CAF50','#F44336'],
        edgecolor='white', linewidth=1.2)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Miss Type Across All Videos', fontsize=11)
mx = max(isolated, persistent) if max(isolated, persistent) > 0 else 1
for i, v in enumerate([isolated, persistent]):
    ax2.text(i, v + mx*0.02, str(v), ha='center', va='bottom', fontsize=11)

plt.tight_layout()
out_path = r'C:\orange_project\results\dino_edge_continuity.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
plt.show()
print(f'  Plot saved -> {out_path}')

# ---
# ## Stage 4 - Visualization: 3 Rows x 5 Columns
#
# Draws the DINO attention heatmap blended onto the frame with blob detections and GT dot markers.
# **Row 1:** 5 random frames &nbsp; **Row 2:** 5 highest F1 &nbsp; **Row 3:** 5 lowest F1
# Green box = TP &nbsp; Red box = FP &nbsp; Yellow dot = GT orange
# Saves to `results/dino_visualization.png`.

# -- Visualization: 3 rows x 5 cols -------------------------------------------
# Row 1: random 5 frames
# Row 2: top 5 highest F1
# Row 3: top 5 lowest F1
#
# For each frame: re-runs DINO attention, overlays heatmap on image,
# draws blob boxes (green=TP, red=FP) and GT dot markers (yellow).

import pickle, random, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open(r'C:\orange_project\notebooks\cache\dino_s1_full.pkl', 'rb') as f:
    saved = pickle.load(f)
preds_all = saved['preds_all']   # fname -> int
f1_store  = saved['f1_store']    # fname -> (prec, rec, f1)
THR   = saved['thr']
MIN_A = saved['min_area']

# Only annotated frames (have both preds and F1)
ann_fns = [fn for fn in preds_all if fn in f1_store and fn in dot_lookup]
by_f1   = sorted(ann_fns, key=lambda f: f1_store[f][2])

random.seed(42)
random_5 = random.sample(ann_fns, 5)
best_5   = by_f1[-5:]   # highest F1
worst_5  = by_f1[:5]    # lowest F1

groups = [
    ('Random',    random_5),
    ('Best F1',   best_5),
    ('Worst F1',  worst_5),
]

SZ = D1_SIZE  # 480

def draw_frame_dino(ax, fname):
    img_bgr = cv2.imread(frame_path[fname])
    H, W    = img_bgr.shape[:2]
    gt      = gt_lookup[fname]
    dots    = dot_lookup.get(fname, [])

    # Re-run DINO attention + blob detection
    attn  = d1_attention(img_bgr)
    blobs = d1_blobs(attn, THR, MIN_A)
    pred  = len(blobs)

    # Overlay attention heatmap on resized image
    img_small   = cv2.resize(img_bgr, (SZ, SZ))
    img_rgb     = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    attn_u8     = (attn * 255).astype(np.uint8)
    heatmap     = cv2.applyColorMap(attn_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blend       = (img_rgb * 0.45 + heatmap_rgb * 0.55).astype(np.uint8)
    ax.imshow(blend)

    # Match blobs to GT dots
    sx = SZ / W; sy = SZ / H
    matched_d, matched_b = set(), set()
    for bi, blob in enumerate(blobs):
        x1, y1, x2, y2 = blob['bbox']
        for di, dot in enumerate(dots):
            if x1 <= dot['x']*sx <= x2 and y1 <= dot['y']*sy <= y2:
                matched_d.add(di); matched_b.add(bi)

    # Draw blobs: green = TP, red = FP
    for bi, blob in enumerate(blobs):
        x1, y1, x2, y2 = blob['bbox']
        color = 'lime' if bi in matched_b else 'red'
        ax.add_patch(mpatches.FancyBboxPatch(
            (x1, y1), x2-x1, y2-y1,
            boxstyle='square,pad=0',
            linewidth=1.5, edgecolor=color, facecolor='none'))

    # Draw GT dots: yellow
    for dot in dots:
        ax.add_patch(plt.Circle(
            (dot['x']*sx, dot['y']*sy),
            4, color='yellow', fill=True, linewidth=0, alpha=0.9))

    tp  = len(matched_d)
    fp  = pred - len(matched_b)
    fn  = len(dots) - tp
    _, _, f1 = f1_store.get(fname, (0, 0, 0))
    err      = pred - gt
    err_str  = f'+{err}' if err > 0 else str(err)
    ax.set_title(
        f'GT={gt} Pred={pred} Err={err_str}\nTP={tp} FP={fp} FN={fn} F1={f1:.2f}',
        fontsize=7, pad=3)
    ax.axis('off')

# -- Plot
fig, axes = plt.subplots(3, 5, figsize=(25, 16))
fig.suptitle(
    f'DINO  -  thr={THR}  min_area={MIN_A}  max_area=12000  (480x480)\n'
    'Heatmap = DINO attention  |  Yellow dot = GT orange  |  Green box = TP  |  Red box = FP',
    fontsize=12, fontweight='bold')

row_labels = ['Random', 'Best F1\n(top 5)', 'Worst F1\n(bottom 5)']
for row, (group_name, fnames) in enumerate(groups):
    axes[row, 0].set_ylabel(row_labels[row], fontsize=11,
                            fontweight='bold', rotation=0,
                            labelpad=70, va='center')
    for col, fname in enumerate(fnames):
        draw_frame_dino(axes[row, col], fname)

plt.tight_layout()
out = r'C:\orange_project\results\dino_visualization.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
plt.show()
print(f'Saved -> {out}')

# -- Error Distribution: Scatter + Histogram -----------------------------------
#
# Plot 1 - Scatter: GT count (x) vs absolute error per frame (y)
#   Each dot = one frame. Shows whether high-count frames are harder.
#   Trend line (red) + mean error line (orange) for reference.
#   Pearson r printed in corner.
#
# Plot 2 - Histogram: how many frames fall in each error bin
#   Bins: 0-2 (good)   2-5 (ok)   5-10 (bad)   10+ (very bad)
#   Each bar shows count and % of all frames.

import pickle, os
import numpy as np
import matplotlib.pyplot as plt

# reuse preds_all / f1_store from full-run cell if in scope, else reload
try:
    _ = preds_all
except NameError:
    with open(r'C:\orange_project\notebooks\cache\dino_s1_full.pkl', 'rb') as f:
        _d = pickle.load(f)
    preds_all = _d['preds_all']
    f1_store  = _d['f1_store']

fnames = [f for f in preds_all if f in gt_lookup]
gts    = np.array([gt_lookup[f]   for f in fnames])
preds  = np.array([preds_all[f]   for f in fnames])
errs   = np.abs(preds - gts)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(
    f'DINO  —  Error Distribution  ({len(fnames):,} frames)',
    fontsize=13, fontweight='bold')

# ── Plot 1: GT count vs absolute error ────────────────────────────────────────
ax = axes[0]
ax.scatter(gts, errs, alpha=0.12, s=5, color='#1976D2', rasterized=True,
           label='frame')

z  = np.polyfit(gts, errs, 1)
xs = np.linspace(gts.min(), gts.max(), 300)
ax.plot(xs, np.poly1d(z)(xs), color='red', linewidth=2.0,
        label=f'Trend  slope={z[0]:.3f}')
ax.axhline(errs.mean(), color='#FF8F00', linewidth=1.5, linestyle='--',
           label=f'Mean error = {errs.mean():.2f}')

corr = float(np.corrcoef(gts, errs)[0, 1])
ax.text(0.97, 0.97, f'r = {corr:.3f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

ax.set_xlabel('GT Count  (ground truth oranges per frame)', fontsize=11)
ax.set_ylabel('Absolute Error  |pred − GT|', fontsize=11)
ax.set_title('GT Count vs Absolute Error per Frame', fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# ── Plot 2: Error bin histogram ────────────────────────────────────────────────
ax2 = axes[1]
bins       = [0, 2, 5, 10, int(errs.max()) + 1]
bin_labels = ['0 – 2\n(good)', '2 – 5\n(ok)', '5 – 10\n(bad)', '10+\n(very bad)']
colors     = ['#43A047', '#FDD835', '#FB8C00', '#E53935']
counts, _  = np.histogram(errs, bins=bins)
pcts       = counts / len(errs) * 100

bars = ax2.bar(bin_labels, counts, color=colors, edgecolor='white',
               linewidth=1.3, width=0.6)
for bar, cnt, pct in zip(bars, counts, pcts):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(counts) * 0.018,
             f'{cnt:,}\n({pct:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_xlabel('Absolute Error Bin', fontsize=11)
ax2.set_ylabel('Number of Frames', fontsize=11)
ax2.set_title('How Many Frames Land in Each Error Bin?', fontsize=11)
ax2.set_ylim(0, max(counts) * 1.22)

# ── Text summary ───────────────────────────────────────────────────────────────
print(f'DINO — Error Distribution  ({len(fnames):,} frames)')
print('-' * 48)
for lbl, cnt, pct in zip(['0-2  (good)    ', '2-5  (ok)      ',
                           '5-10 (bad)     ', '10+  (very bad)'],
                          counts, pcts):
    print(f'  {lbl}  {cnt:>6,}  ({pct:5.1f}%)  {"█" * int(pct / 2)}')
print('-' * 48)
print(f'  Mean error    : {errs.mean():.2f}')
print(f'  Median error  : {float(np.median(errs)):.2f}')
print(f'  Max error     : {int(errs.max())}')
if corr > 0.2:
    verdict = 'Frames with more oranges tend to have larger errors'
elif corr > 0.05:
    verdict = 'Weak trend — count is a minor factor'
else:
    verdict = 'GT count is not a strong predictor of error'
print(f'  Corr(GT, err) : {corr:.3f}  —  {verdict}')

plt.tight_layout()
out = r'C:\orange_project\results\dino_error_distribution.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
plt.show()
print(f'\nSaved -> {out}')
