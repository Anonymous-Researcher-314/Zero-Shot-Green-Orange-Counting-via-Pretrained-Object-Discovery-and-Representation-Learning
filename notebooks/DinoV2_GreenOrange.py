# 03_dinov2.py
# DINOv2 ViT-S/14 (Meta AI 2023) - Full-image K-Means clustering
# Strategy: Full image resized to 392x392 (NO tiling, NO sliding window)
# Best config: k=6  min_area=400  max_area=12000  target=5
# Tree mapping: Tree_483->Tree_01  Tree_484->Tree_02  Tree_490->Tree_03
#               Tree_NN01->Tree_04  Tree_NN02->Tree_05  Tree_2116->Tree_06
#               Tree_4737->Tree_07  Tree_NN03->Tree_08  Tree_NN04->Tree_09
#               Tree_NN05->Tree_10
# NOTE: Filenames on disk (e.g. 483_Vid 01_F001.jpg) are unchanged.

import sys
sys.path.insert(0, r'C:\orange_project\notebooks')
from shared import load_cache, mae, rmse, bias, within_n, compute_f1, tp_mae, box_dot_match

data          = load_cache()
gt_lookup     = data['gt_lookup']
dot_lookup    = data['dot_lookup']
cal_frames    = data['cal_frames']
test_frames   = data['test_frames']
sweep_frames  = data['sweep_frames']
video_groups  = data['video_groups']
gt_df         = data['gt_df']
tree_summary  = data['tree_summary']
cohorts       = data['cohorts']

print(f'Cal frames   : {len(cal_frames):,}')
print(f'Test frames  : {len(test_frames):,}')
print(f'Sweep frames : {len(sweep_frames)}')

import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn', 'scipy', 'scikit-image'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print('scikit-learn, scipy, scikit-image ready')

import os, glob, pickle, time
import numpy as np
import cv2
import torch
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
S14_SIZE  = 392
S14_GRID  = 28
S14_PX    = 14
S14_TARGET   = 5
S14_MAX_AREA = 12000

S14_MEAN = torch.tensor([0.485,0.456,0.406],device=DEVICE).view(1,3,1,1)
S14_STD  = torch.tensor([0.229,0.224,0.225],device=DEVICE).view(1,3,1,1)

if 'model_s' not in dir() or model_s is None:
    print('Loading DINOv2 ViT-S/14 ...')
    model_s = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
    model_s.to(DEVICE).eval()
    print(f'ViT-S/14 loaded  (embed_dim={model_s.embed_dim})')
else:
    print('ViT-S/14 already loaded -- skipping')

if 'frame_path' not in dir() or not frame_path:
    FRAMES_ROOT = r'C:\orange_project\frames'
    frame_path = {}
    for p in glob.glob(os.path.join(FRAMES_ROOT,'**','*.jpg'),recursive=True):
        frame_path[os.path.basename(p)] = p
    print(f'Indexed {len(frame_path):,} frames')

def s14_extract(img_bgr):
    resized = cv2.resize(img_bgr, (S14_SIZE, S14_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t   = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(DEVICE)
    t   = (t - S14_MEAN) / S14_STD
    with torch.no_grad():
        feats = model_s.get_intermediate_layers(t, n=1)[0]
    return feats[0].cpu().numpy()

def s14_count(feats, n_clusters, min_area, max_area=S14_MAX_AREA, target=S14_TARGET):
    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = km.fit_predict(feats).reshape(S14_GRID, S14_GRID)

    best_score, best_mask = -1, None
    for c in range(n_clusters):
        mask = (labels == c).astype(np.uint8)
        lbl, n = ndi.label(mask)
        if n == 0: continue
        sizes  = [int((lbl==i).sum()) for i in range(1, n+1)]
        avg_sz = np.mean(sizes)
        score  = 1.0 / (1.0 + abs(avg_sz - target))
        if score > best_score:
            best_score = score
            best_mask  = mask

    if best_mask is None:
        return 0, []

    lbl, n_obj = ndi.label(best_mask)
    blobs = []
    for i in range(1, n_obj+1):
        region = np.where(lbl == i)
        y1 = int(region[0].min()) * S14_PX
        y2 = int(region[0].max()) * S14_PX + S14_PX - 1
        x1 = int(region[1].min()) * S14_PX
        x2 = int(region[1].max()) * S14_PX + S14_PX - 1
        area = (y2 - y1) * (x2 - x1)
        if min_area <= area <= max_area:
            blobs.append({'bbox': [x1, y1, x2, y2], 'area': area})
    return len(blobs), blobs

def s14_dot_f1(blobs, dots, orig_w, orig_h):
    if not dots and not blobs: return 1.0,1.0,1.0
    if not blobs: return 0.0,0.0,0.0
    if not dots:  return 0.0,0.0,0.0
    sx = S14_SIZE / orig_w; sy = S14_SIZE / orig_h
    md, mb = set(), set()
    for bi, blob in enumerate(blobs):
        x1,y1,x2,y2 = blob['bbox']
        for di, dot in enumerate(dots):
            dx = float(dot['x']) * sx; dy = float(dot['y']) * sy
            if x1 <= dx <= x2 and y1 <= dy <= y2:
                md.add(di); mb.add(bi)
    tp=len(md); fp=len(blobs)-len(mb); fn=len(dots)-len(md)
    pre=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
    return pre, rec, 2*pre*rec/(pre+rec+1e-9)

S14_N_CLUSTERS = [4, 5, 6, 7, 8]
S14_MIN_AREAS  = [100, 200, 400, 800]

configs_s14 = [(k, m) for k in S14_N_CLUSTERS for m in S14_MIN_AREAS]

print(f'{len(configs_s14)} configs: {len(S14_N_CLUSTERS)} n_clusters x {len(S14_MIN_AREAS)} min_areas')
print(f'Target fixed at {S14_TARGET}  |  max_area={S14_MAX_AREA}  |  n_init=auto')
print(f'Input: full 4K frame resized to {S14_SIZE}x{S14_SIZE} -- exact study resolution')
print(f'Area: bounding box in 392x392 pixel space -- exact study metric\n')

def print_summary_s14(label, accum, gts_so_far):
    g=np.array(gts_so_far); rows=[]
    for cfg,s in accum.items():
        p=np.array(s['preds'])
        rows.append((np.mean(np.abs(p-g)), cfg, np.mean(p-g),
                     np.mean(np.abs(p-g)<=2)*100,
                     np.mean(s['prec']), np.mean(s['rec']), np.mean(s['f1'])))
    rows.sort()
    print('\n' + '='*72)
    print(f'  {label}   n={len(g)}   GT mean={g.mean():.1f}')
    print(f'  {"k":>2} {"mina":>5} | {"MAE":>6} {"Bias":>6} {"W+-2%":>6} | {"Prec":>5} {"Rec":>5} {"F1":>5}')
    print('  ' + '-'*68)
    for (mae_val,cfg,bia,w2,pre,rec,f1) in rows:
        k,m=cfg
        print(f'  {k:2d} {m:5d} | {mae_val:6.2f} {bia:+6.2f} {w2:6.1f}% | {pre:5.3f} {rec:5.3f} {f1:5.3f}')
    print('='*72 + '\n')

print(f'Running {len(sweep_frames)} frames -- evaluation every 10 frames\n')
accum_s14 = {cfg:{'preds':[],'prec':[],'rec':[],'f1':[]} for cfg in configs_s14}
gts_s14=[]; t_start=time.time()

for i, fname in enumerate(sweep_frames):
    img  = cv2.imread(frame_path[fname])
    H, W = img.shape[:2]
    gt   = gt_lookup[fname]
    dots = dot_lookup.get(fname, [])
    gts_s14.append(gt)

    t0    = time.time()
    feats = s14_extract(img)
    t_gpu = time.time()-t0

    t1 = time.time()
    for (k, min_a) in configs_s14:
        cnt, blobs = s14_count(feats, k, min_a)
        pre,rec,f1 = s14_dot_f1(blobs, dots, W, H)
        accum_s14[(k,min_a)]['preds'].append(cnt)
        accum_s14[(k,min_a)]['prec'].append(pre)
        accum_s14[(k,min_a)]['rec'].append(rec)
        accum_s14[(k,min_a)]['f1'].append(f1)
    t_cpu = time.time()-t1

    elapsed=time.time()-t_start; eta=elapsed/(i+1)*(len(sweep_frames)-i-1)
    print(f'  [{i+1:2d}/90] {fname:<42}  GT={gt:3d}  '
          f'gpu={t_gpu:.1f}s cpu={t_cpu:.1f}s  ETA {eta/60:.0f}m')

    if (i+1)%10==0:
        print_summary_s14(f'After frame {i+1}/90', accum_s14, gts_s14)

print_summary_s14('FINAL -- all 90 frames', accum_s14, gts_s14)

with open(r'C:\orange_project\notebooks\cache\dinov2_s14_90.pkl','wb') as f:
    pickle.dump({'accum':accum_s14,'gts':gts_s14,
                 'sweep_frames':sweep_frames,'configs':configs_s14},f)
print('Saved -> cache/dinov2_s14_90.pkl')

import os, pickle, time
import numpy as np
from datetime import datetime, timedelta

OUT_PATH         = r'C:\orange_project\results\dinov2_full_results.pkl'
CHECKPOINT_EVERY = 50
BEST_K           = 6
BEST_MIN_A       = 400
BEST_MAX_A       = 12000
BEST_TARGET      = 5

all_frames = list(frame_path.keys())
annotated  = set(fn for fn in all_frames if fn in dot_lookup)

def print_summary(results_dict):
    fns = list(results_dict.keys())
    p   = np.array([results_dict[fn]['pred'] for fn in fns])
    g   = np.array([results_dict[fn]['gt']   for fn in fns])
    f1s = [results_dict[fn]['f1'] for fn in fns]
    mae_v  = np.mean(np.abs(p - g))
    bias_v = np.mean(p - g)
    w2     = np.mean(np.abs(p - g) <= 2) * 100
    rmse_v = np.sqrt(np.mean((p - g) ** 2))
    print('=' * 58)
    print('  DINOv2 - Full Dataset Results')
    print(f'  Config : k={BEST_K}  min_area={BEST_MIN_A}  target={BEST_TARGET}')
    print(f'  Frames : {len(fns):,}  |  Annotated: {len(annotated):,}')
    print('-' * 58)
    print(f'  MAE    = {mae_v:.2f}')
    print(f'  Bias   = {bias_v:+.2f}')
    print(f'  W +- 2  = {w2:.1f} %')
    print(f'  RMSE   = {rmse_v:.2f}')
    print(f'  F1     = {np.mean(f1s):.3f}')
    print('=' * 58)

if os.path.exists(OUT_PATH):
    with open(OUT_PATH, 'rb') as f:
        saved = pickle.load(f)
    if len(saved.get('results', {})) == len(all_frames):
        print(f'Results already saved ({len(all_frames):,} frames). Loading...\n')
        print_summary(saved['results'])
        results = saved['results']
        raise SystemExit

results = {}
if os.path.exists(OUT_PATH):
    with open(OUT_PATH, 'rb') as f:
        saved = pickle.load(f)
    results = saved.get('results', {})
    print(f'Checkpoint found - {len(results):,}/{len(all_frames):,} done. Resuming...')
else:
    print(f'Starting from scratch - {len(all_frames):,} frames')

remaining    = [fn for fn in all_frames if fn not in results]
already_done = len(all_frames) - len(remaining)

print(f'Frames left  : {len(remaining):,}')
print(f'Est. time    : ~{len(remaining)*0.5/60:.0f} min')
print(f'Checkpoint   : every {CHECKPOINT_EVERY} frames\n')
print('-' * 58)

t0 = time.time()

for idx, fname in enumerate(remaining):

    img = cv2.imread(frame_path.get(fname, ''))
    if img is None:
        results[fname] = {'pred': 0, 'gt': gt_lookup[fname], 'prec': 0.0, 'rec': 0.0, 'f1': 0.0}
    else:
        H, W           = img.shape[:2]
        gt             = gt_lookup[fname]
        dots           = dot_lookup.get(fname, [])
        feats          = s14_extract(img)
        pred, blobs    = s14_count(feats, BEST_K, BEST_MIN_A, BEST_MAX_A, BEST_TARGET)
        prec, rec, f1  = s14_dot_f1(blobs, dots, W, H)
        results[fname] = {'pred': pred, 'gt': gt, 'prec': prec, 'rec': rec, 'f1': f1}

    if (idx + 1) % CHECKPOINT_EVERY == 0 or (idx + 1) == len(remaining):
        el         = time.time() - t0
        done_now   = idx + 1
        rate       = el / done_now
        eta_s      = rate * (len(remaining) - done_now)
        finish_dt  = datetime.now() + timedelta(seconds=eta_s)
        total_done = already_done + done_now
        pct        = total_done / len(all_frames) * 100

        print(f'  [{total_done:5d}/{len(all_frames)}]  {pct:5.1f}%  |  '
              f'{rate:.2f} s/frame  |  '
              f'elapsed {el/60:.1f} m  |  '
              f'ETA {eta_s/60:.0f} m  |  '
              f'finish ~{finish_dt.strftime("%H:%M")}')

        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, 'wb') as f:
            pickle.dump({'results': results,
                         'config': {'k': BEST_K, 'min_area': BEST_MIN_A,
                                    'max_area': BEST_MAX_A, 'target': BEST_TARGET}}, f)

print()
print_summary(results)
print(f'\n  Total time : {(time.time()-t0)/60:.1f} min')
print(f'  Saved      -> {OUT_PATH}')

import numpy as np

print()
print('=' * 75)
print('  DINOv2 - Per-Video Breakdown  (all 10,577 frames)')
print(f'  Config : k={BEST_K}   min_area={BEST_MIN_A}   target={BEST_TARGET}')
print('=' * 75)
print(f'  {"Video":<24}  {"n":>5}  {"GT avg":>7}  {"MAE":>7}  {"Bias":>7}  {"W+-2%":>6}  {"RMSE":>7}  {"F1":>6}')
print('  ' + '-' * 71)

_vid_rows_v2 = []
for _vid_id, _fnames in sorted(video_groups.items()):
    _fns = [f for f in _fnames if f in results and f in gt_lookup]
    if not _fns:
        continue
    _ps  = np.array([results[f]['pred'] for f in _fns])
    _gs  = np.array([results[f]['gt']   for f in _fns])
    _f1v = [results[f]['f1'] for f in _fns]
    _vid_rows_v2.append({
        'vid':    _vid_id,
        'n':      len(_fns),
        'gt_avg': float(np.mean(_gs)),
        'mae':    float(np.mean(np.abs(_ps - _gs))),
        'bias':   float(np.mean(_ps - _gs)),
        'w2':     float(np.mean(np.abs(_ps - _gs) <= 2) * 100),
        'rmse':   float(np.sqrt(np.mean((_ps - _gs) ** 2))),
        'f1':     float(np.mean(_f1v)),
    })

_vid_rows_v2.sort(key=lambda r: r['mae'])
for r in _vid_rows_v2:
    print(f'  {r["vid"]:<24}  {r["n"]:>5}  {r["gt_avg"]:>7.1f}  '
          f'{r["mae"]:>7.2f}  {r["bias"]:>+7.2f}  {r["w2"]:>5.1f}%  '
          f'{r["rmse"]:>7.2f}  {r["f1"]:>6.3f}')
print('=' * 75)

import pickle, re, os
import numpy as np
import matplotlib.pyplot as plt

MODEL = 'DINOv2 ViT-S/14'

with open(r'C:\orange_project\results\dinov2_full_results.pkl', 'rb') as f:
    _saved = pickle.load(f)
_res = _saved['results']

def get_pred(fn): return _res[fn]['pred'] if fn in _res else 0
def get_f1(fn):   return _res[fn]['f1']   if fn in _res else 0.0

IMG_W, IMG_H = 3840, 2160
EDGE_FRAC    = 0.15
EX           = IMG_W * EDGE_FRAC
EY           = IMG_H * EDGE_FRAC

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

for vid_id, fnames in video_groups.items():
    fns = [f for f in fnames if f in gt_lookup and f in _res]
    if len(fns) < 3:
        continue
    fns.sort(key=frame_num)
    errs    = [abs(get_pred(f) - gt_lookup[f]) for f in fns]
    is_miss = [e > MISS_THRESH for e in errs]
    n_iso = n_pers = 0
    for i in range(1, len(is_miss) - 1):
        if is_miss[i]:
            if not is_miss[i-1] and not is_miss[i+1]:
                n_iso  += 1
            else:
                n_pers += 1
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
        [isolated, persistent], color=['#4CAF50','#F44336'],
        edgecolor='white', linewidth=1.2)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Miss Type Across All Videos', fontsize=11)
mx = max(isolated, persistent) if max(isolated, persistent) > 0 else 1
for i, v in enumerate([isolated, persistent]):
    ax2.text(i, v + mx*0.02, str(v), ha='center', va='bottom', fontsize=11)

plt.tight_layout()
out_path = r'C:\orange_project\results\dinov2_edge_continuity.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
plt.show()
print(f'  Plot saved -> {out_path}')

import pickle, random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open(r'C:\orange_project\results\dinov2_full_results.pkl', 'rb') as f:
    saved = pickle.load(f)
results = saved['results']

all_fnames = list(results.keys())
by_f1      = sorted(all_fnames, key=lambda f: results[f]['f1'])

random_5 = random.sample(all_fnames, 5)
best_5   = by_f1[-5:]
worst_5  = by_f1[:5]

groups = [
    ('Random',        random_5),
    ('Best F1',       best_5),
    ('Worst F1',      worst_5),
]

SZ           = 392
BEST_K       = 6
BEST_MIN_A   = 400
BEST_MAX_A   = 12000
BEST_TARGET  = 5

def draw_frame(ax, fname, row_label=None):
    img_bgr     = cv2.imread(frame_path[fname])
    H, W        = img_bgr.shape[:2]
    gt          = results[fname]['gt']
    dots        = dot_lookup.get(fname, [])

    feats       = s14_extract(img_bgr)
    pred, blobs = s14_count(feats, BEST_K, BEST_MIN_A, BEST_MAX_A, BEST_TARGET)

    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (SZ, SZ)), cv2.COLOR_BGR2RGB)
    sx = SZ / W;  sy = SZ / H

    matched_d, matched_b = set(), set()
    for bi, blob in enumerate(blobs):
        x1,y1,x2,y2 = blob['bbox']
        for di, dot in enumerate(dots):
            if x1 <= dot['x']*sx <= x2 and y1 <= dot['y']*sy <= y2:
                matched_d.add(di); matched_b.add(bi)

    ax.imshow(img_rgb)

    for bi, blob in enumerate(blobs):
        x1,y1,x2,y2 = blob['bbox']
        color = 'lime' if bi in matched_b else 'red'
        ax.add_patch(mpatches.FancyBboxPatch(
            (x1, y1), x2-x1, y2-y1,
            boxstyle='square,pad=0',
            linewidth=1.5, edgecolor=color, facecolor='none'))

    for dot in dots:
        ax.add_patch(plt.Circle(
            (dot['x']*sx, dot['y']*sy),
            4, color='yellow', fill=True, linewidth=0, alpha=0.9))

    tp  = len(matched_d)
    fp  = pred - len(matched_b)
    fn  = len(dots) - tp
    f1  = results[fname]['f1']
    err = pred - gt
    err_str = f'+{err}' if err > 0 else str(err)

    ax.set_title(
        f'GT={gt} Pred={pred} Err={err_str}\nTP={tp} FP={fp} FN={fn} F1={f1:.2f}',
        fontsize=7, pad=3)
    ax.axis('off')

fig, axes = plt.subplots(3, 5, figsize=(25, 16))
fig.suptitle(
    'DINOv2  -  k=6  target=5  min_area=400  max_area=12000\n'
    'Yellow = GT orange  |  Green box = TP  |  Red box = FP',
    fontsize=12, fontweight='bold')

row_labels = ['Random', 'Best F1\n(top 5)', 'Worst F1\n(bottom 5)']

for row, (group_name, fnames) in enumerate(groups):
    axes[row, 0].set_ylabel(row_labels[row], fontsize=11,
                            fontweight='bold', rotation=0,
                            labelpad=70, va='center')
    for col, fname in enumerate(fnames):
        draw_frame(axes[row, col], fname)

plt.tight_layout()
out = r'C:\orange_project\results\dinov2_visualization.png'
plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
plt.show()
print(f'Saved -> {out}')

