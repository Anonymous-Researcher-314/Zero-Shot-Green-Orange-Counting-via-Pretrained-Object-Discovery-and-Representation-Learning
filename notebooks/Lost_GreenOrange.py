# 02_lost.py
# LOST (Localizing Objects with Self-Supervised Transformers) - BMVC 2021
# Best config: win=128  stride=42  nms=0.25  (480x480 working resolution)
# Tree mapping: Tree_483->Tree_01  Tree_484->Tree_02  Tree_490->Tree_03
#               Tree_NN01->Tree_04  Tree_NN02->Tree_05  Tree_2116->Tree_06
#               Tree_4737->Tree_07  Tree_NN03->Tree_08  Tree_NN04->Tree_09
#               Tree_NN05->Tree_10
# NOTE: Filenames on disk (e.g. 483_Vid 01_F001.jpg) are unchanged.

import sys, os, glob, time, pickle
sys.path.insert(0, r'C:\orange_project\notebooks')
sys.path.insert(0, r'C:\orange_project\model_zoo')

from shared import load_cache
from helpers import box_dot_match, f1_from_counts, FRAMES_ROOT, IMG_W, IMG_H

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.ops import nms as tv_nms
from PIL import Image as PILImage
from datetime import timedelta

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU    : {torch.cuda.get_device_name(0)}')

WORK_SIZE  = 480
PATCH_SIZE = 8
IMG_SIZE   = 224

DOT_SX = WORK_SIZE / IMG_W
DOT_SY = WORK_SIZE / IMG_H

print(f'\nOriginal frame : {IMG_W}x{IMG_H}')
print(f'Work size      : {WORK_SIZE}x{WORK_SIZE}')
print(f'Dot scale x    : /{1/DOT_SX:.1f}   dot scale y : /{1/DOT_SY:.1f}')
print(f'DINO crop      : {IMG_SIZE}x{IMG_SIZE} per window')

print('Loading DINO ViT-S/8 (from cache)...')
dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8',
                      pretrained=True, verbose=False)
dino.eval().to(DEVICE)

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print(f'[OK] DINO ViT-S/8  |  heads={dino.blocks[-1].attn.num_heads}')

data         = load_cache()
gt_lookup    = data['gt_lookup']
dot_lookup   = data['dot_lookup']
sweep_frames = data['sweep_frames']
video_groups = data['video_groups']

frame_path = {}
for p in glob.glob(os.path.join(FRAMES_ROOT, '**', '*.jpg'), recursive=True):
    frame_path[os.path.basename(p)] = p

print(f'\nSweep frames : {len(sweep_frames)}')
print(f'Indexed      : {len(frame_path):,} frames')
print(f'Videos       : {len(video_groups)}')
print(f'xlsx GT      : {sum(1 for f in sweep_frames if f in gt_lookup)}')
print(f'dot  GT      : {sum(1 for f in sweep_frames if f in dot_lookup)}')

def get_dino_features(img_pil):
    """PIL crop (any size) -> DINO patch features (784, 384) + grid size."""
    t = transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = dino.get_intermediate_layers(t, n=1)[0]
    feats = out[0, 1:, :].cpu()
    grid  = IMG_SIZE // PATCH_SIZE
    return feats, grid

def lost_localize(feats, grid):
    """
    LOST seed-expansion - identical to original notebook.
    Returns (x0, y0, x1, y1) in patch-grid coords, or None.
    """
    feats_norm = F.normalize(feats, dim=-1)
    sim_matrix = (feats_norm @ feats_norm.T).numpy()

    avg_sim  = sim_matrix.mean(axis=1)
    seed_idx = int(np.argmin(avg_sim))

    seed_sim = sim_matrix[seed_idx]
    thresh   = np.percentile(seed_sim, 60)
    fg_mask  = (seed_sim >= thresh).reshape(grid, grid)

    rows = np.any(fg_mask, axis=1)
    cols = np.any(fg_mask, axis=0)
    if not rows.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return (c0, r0, c1, r1)

def precompute_raw_boxes(img_pil, win_size, stride,
                         min_area=400, max_area=15000):
    """
    Slide window over img_pil (WORK_SIZExWORK_SIZE PIL image).
    Runs DINO + LOST on every crop.
    Returns (raw_boxes, raw_scores) - lists of boxes BEFORE NMS.
    Box coords are in WORK_SIZE pixel space.
    Area filter identical to original notebook: 400-15000.
    """
    orig_w, orig_h = img_pil.size
    raw_boxes  = []
    raw_scores = []
    grid = IMG_SIZE // PATCH_SIZE

    for y in range(0, orig_h - win_size + 1, stride):
        for x in range(0, orig_w - win_size + 1, stride):
            crop      = img_pil.crop((x, y, x + win_size, y + win_size))
            feats, _  = get_dino_features(crop)
            box_patch = lost_localize(feats, grid)
            if box_patch is None:
                continue

            px0, py0, px1, py1 = box_patch
            scale = win_size / grid
            bx0 = x + int(px0 * scale)
            by0 = y + int(py0 * scale)
            bx1 = x + int((px1 + 1) * scale)
            by1 = y + int((py1 + 1) * scale)

            area = (bx1 - bx0) * (by1 - by0)
            if area < min_area or area > max_area:
                continue

            raw_boxes.append([bx0, by0, bx1, by1])
            raw_scores.append(1.0 / (area + 1))

    return raw_boxes, raw_scores

def apply_nms(raw_boxes, raw_scores, nms_thresh):
    """Apply NMS to raw boxes. Returns (count, boxes_xyxy numpy)."""
    if not raw_boxes:
        return 0, np.zeros((0, 4))
    boxes_t  = torch.tensor(raw_boxes,  dtype=torch.float32)
    scores_t = torch.tensor(raw_scores, dtype=torch.float32)
    keep     = tv_nms(boxes_t, scores_t, nms_thresh)
    return len(keep), boxes_t[keep].numpy()

print(f'Windows per frame at {WORK_SIZE}x{WORK_SIZE}:')
for win in [96, 112, 128]:
    for st in [win // 2, win // 3]:
        nx = len(range(0, WORK_SIZE - win + 1, st))
        ny = len(range(0, WORK_SIZE - win + 1, st))
        print(f'  win={win:3d}  stride={st:3d}  ->  {nx}x{ny} = {nx*ny} windows')
print('\n[OK] LOST functions ready')

NMS_THRESHOLDS = [0.25, 0.35]
MIN_AREA = 400
MAX_AREA = 15000

all_results = {}

for win_size in [96, 112, 128]:
    for stride in [win_size // 2, win_size // 3]:

        nx = len(range(0, WORK_SIZE - win_size + 1, stride))
        ny = len(range(0, WORK_SIZE - win_size + 1, stride))
        print(f'\n{"="*65}')
        print(f'  win={win_size}  stride={stride}  '
              f'({nx}x{ny}={nx*ny} windows/frame)  NMS={NMS_THRESHOLDS}')
        print(f'{"="*65}')

        t_start = time.time()

        all_raw = []

        for fname in sweep_frames:
            img_bgr = cv2.imread(frame_path[fname])
            if img_bgr is None:
                all_raw.append(([], []))
                continue
            small   = cv2.resize(img_bgr, (WORK_SIZE, WORK_SIZE))
            img_pil = PILImage.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            raw_boxes, raw_scores = precompute_raw_boxes(
                img_pil, win_size, stride, MIN_AREA, MAX_AREA
            )
            all_raw.append((raw_boxes, raw_scores))

        print(f'  GPU done in {timedelta(seconds=int(time.time()-t_start))}')

        for nms_t in NMS_THRESHOLDS:

            preds_xlsx = []
            gts_xlsx   = []
            tp_run = fp_run = fn_run = 0

            for i, fname in enumerate(sweep_frames):
                raw_boxes, raw_scores = all_raw[i]
                pred, boxes = apply_nms(raw_boxes, raw_scores, nms_t)

                if fname in gt_lookup:
                    preds_xlsx.append(pred)
                    gts_xlsx.append(gt_lookup[fname])

                if fname in dot_lookup:
                    scaled_dots = [
                        {'x': d['x'] * DOT_SX, 'y': d['y'] * DOT_SY}
                        for d in dot_lookup[fname]
                    ]
                    tp, fp, fn = box_dot_match(boxes, scaled_dots)
                    tp_run += tp
                    fp_run += fp
                    fn_run += fn

                frames_done = i + 1
                if frames_done % 15 == 0 or frames_done == len(sweep_frames):
                    elapsed = timedelta(seconds=int(time.time() - t_start))
                    tag = '[OK] DONE' if frames_done == len(sweep_frames) else '      '

                    p = np.array(preds_xlsx)
                    g = np.array(gts_xlsx)
                    mae_r  = float(np.mean(np.abs(p - g)))          if len(p) else float('nan')
                    bias_r = float(np.mean(p - g))                   if len(p) else float('nan')
                    w2_r   = float(np.mean(np.abs(p-g) <= 2) * 100) if len(p) else float('nan')
                    _, _, f1_r = f1_from_counts(tp_run, fp_run, fn_run)

                    print(f'  nms={nms_t:.2f}  {tag} [{frames_done:>2}/{len(sweep_frames)}]'
                          f'  {elapsed}  |  '
                          f'MAE={mae_r:5.2f}  Bias={bias_r:+6.2f}  W+-2={w2_r:5.1f}%  |  '
                          f'TP={tp_run:4d}  FP={fp_run:4d}  FN={fn_run:4d}  F1={f1_r:.3f}')

            p = np.array(preds_xlsx)
            g = np.array(gts_xlsx)
            _, _, f1_fin = f1_from_counts(tp_run, fp_run, fn_run)
            cfg_key = f'win={win_size} str={stride} nms={nms_t:.2f}'
            all_results[cfg_key] = {
                'win': win_size, 'stride': stride, 'nms': nms_t,
                'mae':  float(np.mean(np.abs(p - g))),
                'bias': float(np.mean(p - g)),
                'w2':   float(np.mean(np.abs(p - g) <= 2) * 100),
                'tp': tp_run, 'fp': fp_run, 'fn': fn_run, 'f1': f1_fin,
            }

print('\n\n' + '='*68)
print('  SWEEP SUMMARY - all 12 configs ranked by MAE')
print('='*68)
print(f'  {"Config":<32} {"MAE":>6} {"Bias":>7} {"W+-2":>7} {"F1":>7}')
print(f'  {"-"*60}')
for cfg, r in sorted(all_results.items(), key=lambda x: x[1]['mae']):
    print(f'  {cfg:<32} {r["mae"]:>6.3f} {r["bias"]:>+7.3f} {r["w2"]:>6.1f}% {r["f1"]:>7.3f}')

best_cfg = min(all_results, key=lambda k: all_results[k]['mae'])
best     = all_results[best_cfg]
print(f'\n  [OK] Best : {best_cfg}')
print(f'     MAE={best["mae"]:.3f}  Bias={best["bias"]:+.3f}'
      f'  W+-2={best["w2"]:.1f}%  F1={best["f1"]:.3f}')

import os, pickle

os.makedirs(r'C:\orange_project\notebooks\cache', exist_ok=True)
out_path = r'C:\orange_project\notebooks\cache\lost_sweep_90.pkl'

with open(out_path, 'wb') as f:
    pickle.dump({
        'all_results':  all_results,
        'sweep_frames': sweep_frames,
        'best_cfg':     best_cfg,
        'best':         best,
        'work_size':    WORK_SIZE,
        'min_area':     MIN_AREA,
        'max_area':     MAX_AREA,
    }, f)

print(f'Saved -> {out_path}')
print(f'\nBest config:  {best_cfg}')
print(f'  MAE={best["mae"]:.3f}  Bias={best["bias"]:+.3f}  W+-2={best["w2"]:.1f}%  F1={best["f1"]:.3f}')

import os, pickle, time
import numpy as np
from datetime import datetime, timedelta

FULL_PKL         = r'C:\orange_project\results\lost_full_results.pkl'
CHECKPOINT_EVERY = 100

BEST_WIN    = 128
BEST_STRIDE = 42
BEST_NMS    = 0.25

print(f'Config  :  win={BEST_WIN}  stride={BEST_STRIDE}  nms={BEST_NMS}')

all_frames = list(frame_path.keys())
annotated  = set(fn for fn in all_frames if fn in dot_lookup)
print(f'Frames  :  {len(all_frames):,} total   {len(annotated):,} annotated')

def _print_summary(res, elapsed_s=None):
    fns    = list(res.keys())
    p      = np.array([res[fn]['pred'] for fn in fns])
    g      = np.array([res[fn]['gt']   for fn in fns])
    f1s    = [res[fn]['f1'] for fn in fns]
    mae_v  = np.mean(np.abs(p - g))
    bias_v = np.mean(p - g)
    w2     = np.mean(np.abs(p - g) <= 2) * 100
    rmse_v = np.sqrt(np.mean((p - g) ** 2))
    print('=' * 58)
    print('  LOST - Full Dataset Results')
    print(f'  Config : win={BEST_WIN}  stride={BEST_STRIDE}  nms={BEST_NMS}')
    print(f'  Frames : {len(fns):,}')
    print('-' * 58)
    print(f'  MAE    = {mae_v:.2f}')
    print(f'  Bias   = {bias_v:+.2f}')
    print(f'  W +- 2  = {w2:.1f} %')
    print(f'  RMSE   = {rmse_v:.2f}')
    print(f'  F1     = {np.mean(f1s):.3f}')
    if elapsed_s is not None:
        print(f'  Time   = {elapsed_s/60:.1f} min')
    print('=' * 58)

if os.path.exists(FULL_PKL):
    with open(FULL_PKL, 'rb') as f:
        saved = pickle.load(f)
    if len(saved.get('results', {})) == len(all_frames):
        print(f'\nAlready complete ({len(all_frames):,} frames). Loading...\n')
        results = saved['results']
        _print_summary(results)
        raise SystemExit

results = {}
if os.path.exists(FULL_PKL):
    with open(FULL_PKL, 'rb') as f:
        saved = pickle.load(f)
    results = saved.get('results', {})
    print(f'\nCheckpoint found - {len(results):,}/{len(all_frames):,} done. Resuming...')
else:
    print(f'\nNo checkpoint - starting from scratch.')

remaining    = [fn for fn in all_frames if fn not in results]
already_done = len(all_frames) - len(remaining)

n_windows   = len(list(range(0, WORK_SIZE - BEST_WIN + 1, BEST_STRIDE))) ** 2
est_s_frame = n_windows * 0.012
est_total   = len(remaining) * est_s_frame

print(f'Windows/frame : {n_windows}')
print(f'Frames left   : {len(remaining):,}')
print(f'Est. time     : ~{est_total/3600:.1f} h  ({est_total/60:.0f} min)')
print(f'Checkpoint    : every {CHECKPOINT_EVERY} frames\n')
print('-' * 58)

t0 = time.time()

for idx, fname in enumerate(remaining):

    img_bgr = cv2.imread(frame_path.get(fname, ''))
    if img_bgr is None:
        results[fname] = {'pred': 0, 'gt': gt_lookup[fname], 'f1': 0.0}
    else:
        small   = cv2.resize(img_bgr, (WORK_SIZE, WORK_SIZE))
        img_pil = PILImage.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        raw_boxes, raw_scores = precompute_raw_boxes(
            img_pil, BEST_WIN, BEST_STRIDE, MIN_AREA, MAX_AREA)
        pred, boxes = apply_nms(raw_boxes, raw_scores, BEST_NMS)

        f1 = 0.0
        if fname in annotated:
            scaled_dots = [{'x': d['x'] * DOT_SX, 'y': d['y'] * DOT_SY}
                           for d in dot_lookup[fname]]
            tp, fp, fn_c = box_dot_match(boxes, scaled_dots)
            _, _, f1     = f1_from_counts(tp, fp, fn_c)

        results[fname] = {'pred': pred, 'gt': gt_lookup[fname], 'f1': f1}

    total_done = already_done + idx + 1
    show = ((total_done <= 100 and total_done % 10 == 0) or
            (total_done > 100 and total_done % 1000 == 0) or
            (idx + 1) == len(remaining))

    if show:
        el        = time.time() - t0
        rate      = el / (idx + 1)
        eta_s     = rate * (len(remaining) - idx - 1)
        finish_dt = datetime.now() + timedelta(seconds=eta_s)
        pct       = total_done / len(all_frames) * 100
        p_run     = np.array([results[fn]['pred'] for fn in results])
        g_run     = np.array([results[fn]['gt']   for fn in results])
        mae_run   = np.mean(np.abs(p_run - g_run))

        print(f'  [{total_done:5d}/{len(all_frames)}]  {pct:5.1f}%  |  '
              f'{rate:.2f} s/fr  |  '
              f'elapsed {el/60:.1f} m  |  '
              f'ETA {eta_s/60:.0f} m  |  '
              f'finish ~{finish_dt.strftime("%H:%M")}  |  '
              f'MAE={mae_run:.2f}')

    if (idx + 1) % CHECKPOINT_EVERY == 0 or (idx + 1) == len(remaining):
        os.makedirs(os.path.dirname(FULL_PKL), exist_ok=True)
        with open(FULL_PKL, 'wb') as f:
            pickle.dump({'results': results,
                         'config': {'win': BEST_WIN, 'stride': BEST_STRIDE,
                                    'nms': BEST_NMS}}, f)

print()
_print_summary(results, elapsed_s=time.time() - t0)
print(f'\n  Saved -> {FULL_PKL}')

import numpy as np

print()
print('=' * 75)
print('  LOST - Per-Video Breakdown  (all 10,577 frames)')
print(f'  Config : win={BEST_WIN}   stride={BEST_STRIDE}   nms={BEST_NMS}')
print('=' * 75)
print(f'  {"Video":<24}  {"n":>5}  {"GT avg":>7}  {"MAE":>7}  {"Bias":>7}  {"W+-2%":>6}  {"RMSE":>7}  {"F1":>6}')
print('  ' + '-' * 71)

_vid_rows_lost = []
for _vid_id, _fnames in sorted(video_groups.items()):
    _fns = [f for f in _fnames if f in results and f in gt_lookup]
    if not _fns:
        continue
    _ps  = np.array([results[f]['pred'] for f in _fns])
    _gs  = np.array([results[f]['gt']   for f in _fns])
    _f1v = [results[f]['f1'] for f in _fns]
    _vid_rows_lost.append({
        'vid':    _vid_id,
        'n':      len(_fns),
        'gt_avg': float(np.mean(_gs)),
        'mae':    float(np.mean(np.abs(_ps - _gs))),
        'bias':   float(np.mean(_ps - _gs)),
        'w2':     float(np.mean(np.abs(_ps - _gs) <= 2) * 100),
        'rmse':   float(np.sqrt(np.mean((_ps - _gs) ** 2))),
        'f1':     float(np.mean(_f1v)),
    })

_vid_rows_lost.sort(key=lambda r: r['mae'])
for r in _vid_rows_lost:
    print(f'  {r["vid"]:<24}  {r["n"]:>5}  {r["gt_avg"]:>7.1f}  '
          f'{r["mae"]:>7.2f}  {r["bias"]:>+7.2f}  {r["w2"]:>5.1f}%  '
          f'{r["rmse"]:>7.2f}  {r["f1"]:>6.3f}')
print('=' * 75)

import pickle, re, os
import numpy as np
import matplotlib.pyplot as plt

MODEL = 'LOST'

with open(r'C:\orange_project\results\lost_full_results.pkl', 'rb') as f:
    _saved_an = pickle.load(f)
_res = _saved_an['results']

def get_pred(fn): return _res[fn]['pred'] if fn in _res else 0
def get_f1(fn):   return _res[fn]['f1']   if fn in _res else 0.0

with open(r'C:\orange_project\notebooks\cache\video_groups.pkl', 'rb') as f:
    _vg = pickle.load(f)

IMG_W, IMG_H = 3840, 2160
EDGE_FRAC    = 0.15
EX = IMG_W * EDGE_FRAC
EY = IMG_H * EDGE_FRAC

def dot_is_edge(d):
    x, y = float(d['x']), float(d['y'])
    return x < EX or x > IMG_W - EX or y < EY or y > IMG_H - EY

records = []
for fname, dots in dot_lookup.items():
    if fname not in _res or len(dots) == 0:
        continue
    n_edge = sum(1 for d in dots if dot_is_edge(d))
    ef     = n_edge / len(dots)
    f1     = get_f1(fname)
    gt     = gt_lookup.get(fname, 0)
    pred   = get_pred(fname)
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

MISS_THRESH = 5

def frame_num(fname):
    m = re.search(r'_F(\d+)\.jpg$', fname, re.IGNORECASE)
    return int(m.group(1)) if m else 0

isolated = persistent = total_frames = 0
video_rows = []

for vid_id, fnames in _vg.items():
    fns = [f for f in fnames if f in gt_lookup and f in _res]
    if len(fns) < 3:
        continue
    fns.sort(key=frame_num)
    errs    = [abs(get_pred(f) - gt_lookup[f]) for f in fns]
    is_miss = [e > MISS_THRESH for e in errs]
    n_iso = n_pers = 0
    for i in range(1, len(is_miss) - 1):
        if is_miss[i]:
            if not is_miss[i-1] and not is_miss[i+1]: n_iso  += 1
            else:                                       n_pers += 1
    isolated     += n_iso
    persistent   += n_pers
    total_frames += len(fns)
    video_rows.append({'vid': vid_id, 'n': len(fns),
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
        [isolated, persistent], color=['#4CAF50', '#F44336'],
        edgecolor='white', linewidth=1.2)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Miss Type Across All Videos', fontsize=11)
mx = max(isolated, persistent) if max(isolated, persistent) > 0 else 1
for i, v in enumerate([isolated, persistent]):
    ax2.text(i, v + mx*0.02, str(v), ha='center', va='bottom', fontsize=11)

plt.tight_layout()
out_path = r'C:\orange_project\results\lost_edge_continuity.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
plt.show()
print(f'  Plot saved -> {out_path}')

import pickle, random, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image as PILImage

with open(r'C:\orange_project\results\lost_full_results.pkl', 'rb') as f:
    _saved_viz = pickle.load(f)
results_viz = _saved_viz['results']

ann_fns = [fn for fn in results_viz if fn in dot_lookup]
by_f1   = sorted(ann_fns, key=lambda f: results_viz[f]['f1'])

random.seed(42)
random_5 = random.sample(ann_fns, 5)
best_5   = by_f1[-5:]
worst_5  = by_f1[:5]

groups = [('Random', random_5), ('Best F1', best_5), ('Worst F1', worst_5)]

def draw_frame_lost(ax, fname):
    img_bgr = cv2.imread(frame_path[fname])
    H, W    = img_bgr.shape[:2]
    gt      = gt_lookup[fname]
    dots    = dot_lookup.get(fname, [])

    small   = cv2.resize(img_bgr, (WORK_SIZE, WORK_SIZE))
    img_pil = PILImage.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    raw_boxes, raw_scores = precompute_raw_boxes(img_pil, BEST_WIN, BEST_STRIDE, MIN_AREA, MAX_AREA)
    pred, boxes = apply_nms(raw_boxes, raw_scores, BEST_NMS)

    ax.imshow(np.array(img_pil))

    sx = WORK_SIZE / W; sy = WORK_SIZE / H
    scaled_dots = [{'x': d['x']*sx, 'y': d['y']*sy} for d in dots]

    matched_d, matched_b = set(), set()
    for bi, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        for di, dot in enumerate(scaled_dots):
            if x1 <= dot['x'] <= x2 and y1 <= dot['y'] <= y2:
                matched_d.add(di); matched_b.add(bi)

    for bi, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = 'lime' if bi in matched_b else 'red'
        ax.add_patch(mpatches.FancyBboxPatch(
            (x1, y1), x2-x1, y2-y1,
            boxstyle='square,pad=0',
            linewidth=1.5, edgecolor=color, facecolor='none'))

    for dot in scaled_dots:
        ax.add_patch(plt.Circle(
            (dot['x'], dot['y']), 4, color='yellow', fill=True, linewidth=0, alpha=0.9))

    tp      = len(matched_d)
    fp      = pred - len(matched_b)
    fn      = len(dots) - tp
    f1      = results_viz[fname]['f1']
    err     = pred - gt
    err_str = f'+{err}' if err > 0 else str(err)
    ax.set_title(
        f'GT={gt} Pred={pred} Err={err_str}\nTP={tp} FP={fp} FN={fn} F1={f1:.2f}',
        fontsize=7, pad=3)
    ax.axis('off')

fig, axes = plt.subplots(3, 5, figsize=(25, 16))
fig.suptitle(
    'LOST  -  win=128  stride=42  nms=0.25  (480x480)\n'
    'Yellow dot = GT orange  |  Green box = TP  |  Red box = FP',
    fontsize=12, fontweight='bold')

row_labels = ['Random', 'Best F1\n(top 5)', 'Worst F1\n(bottom 5)']
for row, (group_name, fnames) in enumerate(groups):
    axes[row, 0].set_ylabel(row_labels[row], fontsize=11,
                            fontweight='bold', rotation=0,
                            labelpad=70, va='center')
    for col, fname in enumerate(fnames):
        draw_frame_lost(axes[row, col], fname)

plt.tight_layout()
out = r'C:\orange_project\results\lost_visualization.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
plt.show()
print(f'Saved -> {out}')

import pickle, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image as PILImage

try:
    _ = BEST_WIN
except NameError:
    BEST_WIN    = 128
    BEST_STRIDE = 42
    BEST_NMS    = 0.25
    MIN_AREA    = 400
    MAX_AREA    = 15000

try:
    _ = results
except NameError:
    with open(r'C:\orange_project\results\lost_full_results.pkl', 'rb') as f:
        results = pickle.load(f)['results']

scored = []
for fname, r in results.items():
    if fname not in frame_path:
        continue
    err  = abs(r['pred'] - r['gt'])
    pred = r['pred']
    gt   = r['gt']
    scored.append((err, pred, gt, fname))

tier1 = [(e, p, g, f) for e, p, g, f in scored if e <= 2 and p >= 1]
tier2 = [(e, p, g, f) for e, p, g, f in scored if e <= 3 and p >= 1]
tier3 = sorted([(e, p, g, f) for e, p, g, f in scored if p >= 1],
               key=lambda x: x[0])

pool = tier1 if len(tier1) >= 15 else (tier2 if len(tier2) >= 15 else tier3)
pool.sort(key=lambda x: x[0])

n_need = 15
if len(pool) >= n_need:
    step = max(1, len(pool) // n_need)
    good_frames = pool[::step][:n_need]
    if len(good_frames) < n_need:
        good_frames = pool[:n_need]
else:
    good_frames = pool[:n_need]

print(f'Pool breakdown:')
print(f'  Tier 1 (err<=2, pred>=1) : {len(tier1):,} frames')
print(f'  Tier 2 (err<=3, pred>=1) : {len(tier2):,} frames')
print(f'  Tier 3 (pred>=1, any err): {len(tier3):,} frames')
print(f'  Showing {len(good_frames)} frames from {"Tier 1" if pool is tier1 else "Tier 2" if pool is tier2 else "Tier 3"}')
print()
print(f'  {"Frame":<40}  {"GT":>4}  {"Pred":>5}  {"Err":>5}')
print('  ' + '-' * 55)
for err, pred, gt, fname in good_frames:
    sign = '+' if pred - gt > 0 else ''
    print(f'  {fname:<40}  {gt:>4}  {pred:>5}  {sign}{pred-gt:>4}')

def _draw_good_frame(ax, fname, gt_val):
    img_bgr = cv2.imread(frame_path[fname])
    H, W    = img_bgr.shape[:2]
    dots    = dot_lookup.get(fname, [])

    small   = cv2.resize(img_bgr, (WORK_SIZE, WORK_SIZE))
    img_pil = PILImage.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    raw_boxes, raw_scores = precompute_raw_boxes(
        img_pil, BEST_WIN, BEST_STRIDE, MIN_AREA, MAX_AREA)
    pred_count, boxes = apply_nms(raw_boxes, raw_scores, BEST_NMS)

    ax.imshow(np.array(img_pil))

    sx = WORK_SIZE / W
    sy = WORK_SIZE / H
    scaled_dots = [{'x': d['x'] * sx, 'y': d['y'] * sy} for d in dots]

    matched_d, matched_b = set(), set()
    for bi, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        for di, dot in enumerate(scaled_dots):
            if x1 <= dot['x'] <= x2 and y1 <= dot['y'] <= y2:
                matched_d.add(di)
                matched_b.add(bi)

    for bi, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = 'lime' if bi in matched_b else 'red'
        ax.add_patch(mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle='square,pad=0',
            linewidth=1.5, edgecolor=color, facecolor='none'))

    for dot in scaled_dots:
        ax.add_patch(plt.Circle(
            (dot['x'], dot['y']), 4,
            color='yellow', fill=True, linewidth=0, alpha=0.9))

    tp  = len(matched_d)
    fp  = pred_count - len(matched_b)
    fn  = len(dots) - tp
    err = pred_count - gt_val
    err_str = f'+{err}' if err > 0 else str(err)
    ax.set_title(
        f'GT={gt_val}  Pred={pred_count}  Err={err_str}\n'
        f'TP={tp}  FP={fp}  FN={fn}',
        fontsize=7, pad=3)
    ax.axis('off')

rows, cols = 3, 5
fig, axes = plt.subplots(rows, cols, figsize=(25, 16))
fig.suptitle(
    'LOST  —  Best-Result Frames  (lowest |pred − GT|,  pred ≥ 1)\n'
    'Yellow dot = GT orange  |  Green box = TP  |  Red box = FP',
    fontsize=12, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < len(good_frames):
        err, pred, gt, fname = good_frames[i]
        _draw_good_frame(ax, fname, gt)
    else:
        ax.axis('off')

plt.tight_layout()
out = r'C:\orange_project\results\lost_good_frames.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
plt.show()
print(f'\nSaved -> {out}')

import pickle, os
import numpy as np
import matplotlib.pyplot as plt

try:
    _ = results
except NameError:
    with open(r'C:\orange_project\results\lost_full_results.pkl', 'rb') as f:
        results = pickle.load(f)['results']

fnames = list(results.keys())
gts    = np.array([results[f]['gt']   for f in fnames])
preds  = np.array([results[f]['pred'] for f in fnames])
errs   = np.abs(preds - gts)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(
    f'LOST  —  Error Distribution  ({len(fnames):,} frames)',
    fontsize=13, fontweight='bold')

ax = axes[0]
ax.scatter(gts, errs, alpha=0.12, s=5, color='#1E88E5', rasterized=True,
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

print(f'LOST — Error Distribution  ({len(fnames):,} frames)')
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
out = r'C:\orange_project\results\lost_error_distribution.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
plt.show()
print(f'\nSaved -> {out}')
