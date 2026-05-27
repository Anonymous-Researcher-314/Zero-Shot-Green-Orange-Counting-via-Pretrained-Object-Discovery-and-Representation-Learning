# %% [markdown]
# # Zero-Shot Green Orange Detection — Shared Setup
# **Run this notebook once. All other notebooks load from `cache/`.**
#
# | Split | Trees | Purpose |
# |-------|-------|---------|
# | Calibration | Tree_01, Tree_04, Tree_05, Tree_06, Tree_07, Tree_08 | Threshold tuning only |
# | Test        | Tree_02, Tree_03, Tree_09, Tree_10                   | Final honest results  |

# %% [markdown]
# ## 1. Environment Setup

# %%
import os, sys, glob, re, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from collections import defaultdict
import torch
import torchvision
from config import TREE_ID_MAP, TREE_40SEC_RAW_IDS, CAL_RAW_IDS, TEST_RAW_IDS

FRAMES_ROOT = r'C:\orange_project\frames'
ANNOT_DIR   = r'C:\orange_project\annotations'
LABELED_DIR = r'C:\orange_project\labeled'
RESULTS_DIR = r'C:\orange_project\results'
CACHE_DIR   = r'C:\orange_project\notebooks\cache'

IMG_W, IMG_H = 2160, 3840

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,   exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'PyTorch     : {torch.__version__}')
print(f'Torchvision : {torchvision.__version__}')
print(f'Device      : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU         : {torch.cuda.get_device_name(0)}')
    print(f'VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
for name, path in [('frames', FRAMES_ROOT), ('annotations', ANNOT_DIR),
                   ('labeled', LABELED_DIR), ('results', RESULTS_DIR), ('cache', CACHE_DIR)]:
    print(f'  {"OK" if os.path.exists(path) else "MISSING":7} {name}: {path}')

# %% [markdown]
# ## 2. Data Verification

# %%
errors = []

all_frame_paths = sorted(glob.glob(os.path.join(FRAMES_ROOT, '**', '*.jpg'), recursive=True))
all_frame_names = {os.path.basename(p) for p in all_frame_paths}
print(f'[2a] Frames on disk     : {len(all_frame_paths):,}')

xlsx_files = sorted(glob.glob(os.path.join(ANNOT_DIR, '*.xlsx')))
raw_df = pd.concat([pd.read_excel(f) for f in xlsx_files], ignore_index=True)
raw_df['ground_truth_count'] = pd.to_numeric(raw_df['ground_truth_count'], errors='coerce')
print(f'[2b] XLSX rows          : {len(raw_df):,}  (files: {len(xlsx_files)})')
missing_in_frames = raw_df['image_filename'].dropna().apply(lambda f: f not in all_frame_names).sum()
if missing_in_frames:
    errors.append(f'  {missing_in_frames} xlsx filenames not found on disk')
else:
    print(f'     All xlsx filenames found on disk')

json_files = sorted(set(
    glob.glob(os.path.join(LABELED_DIR, '**', '*.json'), recursive=True) +
    glob.glob(os.path.join(LABELED_DIR, '*.json'))
))
_dot_total = 0; _dot_frames = 0; _coord_errors = 0; _temp_dot = {}
for jf in json_files:
    data       = json.load(open(jf, encoding='utf-8'))
    tree       = data['treeId']
    vid_digits = re.search(r'\d+', data['videoId']).group().zfill(2)
    for idx_str, dots in data['annotations'].items():
        frame_num = int(idx_str) + 1
        fname = f'{tree}_Vid {vid_digits}_F{str(frame_num).zfill(3)}.jpg'
        _temp_dot[fname] = dots
        _dot_frames += 1
        for d in dots:
            _dot_total += 1
            if not (0 <= d['x'] <= IMG_W and 0 <= d['y'] <= IMG_H):
                _coord_errors += 1
print(f'[2c] JSON dot files     : {len(json_files)}')
print(f'     Annotated frames   : {_dot_frames:,}')
print(f'     Total dots         : {_dot_total:,}')
if _coord_errors:
    errors.append(f'  {_coord_errors} dots outside image bounds ({IMG_W}x{IMG_H})')
else:
    print(f'     Coordinate bounds  : all within {IMG_W}x{IMG_H}')

if len(all_frame_paths) != len(raw_df):
    errors.append(f'  Frame count mismatch: {len(all_frame_paths)} on disk vs {len(raw_df)} in xlsx')
else:
    print(f'[2d] Frame count match  : {len(all_frame_paths):,}')

nan_count = raw_df['ground_truth_count'].isna().sum()
if nan_count:
    errors.append(f'  {nan_count} NaN values in ground_truth_count')
else:
    print(f'[2e] No NaN counts')

print()
if errors:
    print('VERIFICATION FAILED:')
    for e in errors: print(e)
else:
    print('All verification checks passed — safe to continue.')

# %% [markdown]
# ## 3. Load Ground Truth

# %%
gt_df = raw_df.copy()
gt_df['ground_truth_count'] = gt_df['ground_truth_count'].fillna(0).astype(int)
gt_df['tree']  = gt_df['image_filename'].str.extract(r'^(.+)_Vid')
gt_df['video'] = gt_df['image_filename'].str.extract(r'Vid (\d+)')
gt_df['tree']  = gt_df['tree'].map(TREE_ID_MAP).fillna(gt_df['tree'])

gt_lookup  = dict(zip(gt_df['image_filename'], gt_df['ground_truth_count']))
dot_lookup = _temp_dot

tree_summary = (gt_df.groupby('tree')
                .agg(
                    n_frames      = ('image_filename', 'count'),
                    avg_count     = ('ground_truth_count', 'mean'),
                    min_count     = ('ground_truth_count', 'min'),
                    max_count     = ('ground_truth_count', 'max'),
                    total_xlsx_gt = ('ground_truth_count', 'sum'),
                )
                .round(2)
                .reset_index())

dot_per_tree = defaultdict(int)
for fname, dots in dot_lookup.items():
    raw_id     = re.sub(r'_Vid.*', '', fname)
    display_id = TREE_ID_MAP.get(raw_id, raw_id)
    dot_per_tree[display_id] += len(dots)
tree_summary['total_dots'] = tree_summary['tree'].map(dot_per_tree).fillna(0).astype(int)

print(tree_summary.to_string(index=False))
print(f'\nTotal — frames: {len(gt_df):,}  |  GT total: {gt_df["ground_truth_count"].sum():,}  |  dots: {_dot_total:,}')

# %% [markdown]
# ## 4. Exploratory Data Analysis

# %%
TREES_40SEC_DISPLAY = [TREE_ID_MAP[r] for r in TREE_40SEC_RAW_IDS]
TREES_30SEC_DISPLAY = [v for v in TREE_ID_MAP.values() if v not in TREES_40SEC_DISPLAY]

COHORTS = {
    '30 sec': TREES_30SEC_DISPLAY,
    '40 sec': TREES_40SEC_DISPLAY,
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.hist(gt_df['ground_truth_count'], bins=40, color='steelblue', edgecolor='white')
ax.axvline(gt_df['ground_truth_count'].mean(),   color='red',    linestyle='--',
           label=f'Mean {gt_df["ground_truth_count"].mean():.1f}')
ax.axvline(gt_df['ground_truth_count'].median(), color='orange', linestyle='--',
           label=f'Median {gt_df["ground_truth_count"].median():.1f}')
ax.set_title('Orange Count Distribution (all frames)')
ax.set_xlabel('Oranges per frame'); ax.set_ylabel('Number of frames'); ax.legend()

ax = axes[1]
cohort_color = {t: 'steelblue' for t in COHORTS['30 sec']}
cohort_color.update({t: 'coral' for t in COHORTS['40 sec']})
colors = [cohort_color.get(t, 'gray') for t in tree_summary['tree']]
ax.bar(tree_summary['tree'], tree_summary['avg_count'], color=colors, edgecolor='white')
ax.set_title('Average Orange Count per Tree')
ax.set_xlabel('Tree'); ax.set_ylabel('Avg count per frame')
ax.tick_params(axis='x', rotation=45)
ax.legend(handles=[mpatches.Patch(color='steelblue', label='30 sec'),
                   mpatches.Patch(color='coral',     label='40 sec')])

ax = axes[2]
cohort_data, cohort_labels = [], []
for cname, trees in COHORTS.items():
    data = gt_df[gt_df['tree'].isin(trees)]['ground_truth_count'].values
    cohort_data.append(data); cohort_labels.append(f'{cname}\n(n={len(data):,})')
ax.boxplot(cohort_data, labels=cohort_labels, patch_artist=True,
           boxprops=dict(facecolor='steelblue', alpha=0.6))
ax.set_title('Count Distribution: 30s vs 40s'); ax.set_ylabel('Oranges per frame')

plt.tight_layout()
plt.savefig(os.path.join(CACHE_DIR, 'eda_overview.png'), dpi=120, bbox_inches='tight')
plt.show()

for cname, trees in COHORTS.items():
    sub = gt_df[gt_df['tree'].isin(trees)]['ground_truth_count']
    print(f'{cname}: {len(sub):,} frames | mean {sub.mean():.2f} | median {sub.median():.1f} | max {sub.max()}')

# %% [markdown]
# ## 5. Metrics

# %%
def mae(preds, gts):
    return np.mean([abs(p - g) for p, g in zip(preds, gts)])

def rmse(preds, gts):
    return np.sqrt(np.mean([(p - g) ** 2 for p, g in zip(preds, gts)]))

def bias(preds, gts):
    return np.mean([p - g for p, g in zip(preds, gts)])

def within_n(preds, gts, n=2):
    return np.mean([abs(p - g) <= n for p, g in zip(preds, gts)])

def compute_f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def tp_mae(tp_counts, gt_counts):
    return np.mean([abs(t - g) for t, g in zip(tp_counts, gt_counts)])

def box_dot_match(boxes_px, dots):
    dot_matched = [False] * len(dots)
    tp = fp = 0
    for (x1, y1, x2, y2) in boxes_px:
        hit = False
        for j, d in enumerate(dots):
            if dot_matched[j]: continue
            if x1 <= d['x'] <= x2 and y1 <= d['y'] <= y2:
                dot_matched[j] = True; hit = True; break
        tp += hit; fp += not hit
    fn = sum(1 for m in dot_matched if not m)
    return tp, fp, fn

demo_preds = [12, 9, 11, 14, 8]
demo_gts   = [10, 10, 10, 10, 10]
print(f'MAE      : {mae(demo_preds, demo_gts):.2f}')
print(f'RMSE     : {rmse(demo_preds, demo_gts):.2f}')
print(f'Bias     : {bias(demo_preds, demo_gts):+.2f}')
print(f'Within±2 : {within_n(demo_preds, demo_gts, 2):.0%}')

shared_py_path = r'C:\orange_project\notebooks\shared.py'
shared_code = (
    "import os, pickle\nimport numpy as np\n\n"
    "CACHE_DIR = r'C:\\orange_project\\notebooks\\cache'\n\n"
    "def load_cache():\n"
    "    keys = ['gt_lookup','dot_lookup','cal_frames','test_frames',\n"
    "            'sweep_frames','video_groups','gt_df','tree_summary','cohorts']\n"
    "    out = {}\n"
    "    for k in keys:\n"
    "        with open(os.path.join(CACHE_DIR, f'{k}.pkl'), 'rb') as f:\n"
    "            out[k] = pickle.load(f)\n"
    "    print('Cache loaded:', ', '.join(keys))\n"
    "    return out\n\n"
    "def mae(preds, gts):         return __import__('numpy').mean([abs(p-g) for p,g in zip(preds,gts)])\n"
    "def rmse(preds, gts):        return __import__('numpy').sqrt(__import__('numpy').mean([(p-g)**2 for p,g in zip(preds,gts)]))\n"
    "def bias(preds, gts):        return __import__('numpy').mean([p-g for p,g in zip(preds,gts)])\n"
    "def within_n(preds,gts,n=2): return __import__('numpy').mean([abs(p-g)<=n for p,g in zip(preds,gts)])\n"
    "def compute_f1(tp,fp,fn):\n"
    "    prec=tp/(tp+fp) if (tp+fp)>0 else 0.0\n"
    "    rec=tp/(tp+fn) if (tp+fn)>0 else 0.0\n"
    "    f1=2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0\n"
    "    return prec,rec,f1\n"
    "def tp_mae(tp_counts,gt_counts): return __import__('numpy').mean([abs(t-g) for t,g in zip(tp_counts,gt_counts)])\n"
    "def box_dot_match(boxes_px,dots):\n"
    "    dot_matched=[False]*len(dots); tp=fp=0\n"
    "    for (x1,y1,x2,y2) in boxes_px:\n"
    "        hit=False\n"
    "        for j,d in enumerate(dots):\n"
    "            if dot_matched[j]: continue\n"
    "            if x1<=d['x']<=x2 and y1<=d['y']<=y2:\n"
    "                dot_matched[j]=True; hit=True; break\n"
    "        tp+=hit; fp+=not hit\n"
    "    fn=sum(1 for m in dot_matched if not m)\n"
    "    return tp,fp,fn\n"
)
with open(shared_py_path, 'w', encoding='utf-8') as f:
    f.write(shared_code)
print(f'shared.py written to {shared_py_path}')

# %% [markdown]
# ## 6. Cal / Test Split & Save Cache

# %%
import re as _re

CAL_TREES  = [TREE_ID_MAP[r] for r in CAL_RAW_IDS]
TEST_TREES = [TREE_ID_MAP[r] for r in TEST_RAW_IDS]

all_trees = sorted(gt_df['tree'].unique())
print(f'All trees ({len(all_trees)}): {all_trees}')
missing = set(all_trees) - (set(CAL_TREES) | set(TEST_TREES))
if missing: print(f'WARNING: unassigned trees: {missing}')

def frames_for_trees(tree_list):
    return sorted(gt_df[gt_df['tree'].isin(tree_list)]['image_filename'].tolist())

cal_frames  = frames_for_trees(CAL_TREES)
test_frames = frames_for_trees(TEST_TREES)

print(f'\nCalibration : {len(CAL_TREES)} trees  |  {len(cal_frames):,} frames')
for t in CAL_TREES:
    sub    = gt_df[gt_df['tree'] == t]
    cohort = '30s' if t in COHORTS['30 sec'] else '40s'
    print(f'  [{cohort}] {t:<10} {len(sub):>4} frames  avg {sub["ground_truth_count"].mean():.1f}')
print(f'\nTest        : {len(TEST_TREES)} trees  |  {len(test_frames):,} frames')
for t in TEST_TREES:
    sub    = gt_df[gt_df['tree'] == t]
    cohort = '30s' if t in COHORTS['30 sec'] else '40s'
    print(f'  [{cohort}] {t:<10} {len(sub):>4} frames  avg {sub["ground_truth_count"].mean():.1f}')

def _video_key(fname):  return _re.sub(r'_F\d+\.jpg$', '', fname)
def _frame_num(fname):
    m = _re.search(r'_F(\d+)\.jpg$', fname)
    return int(m.group(1)) if m else 0

all_fnames_sorted = sorted(gt_lookup.keys(), key=lambda f: (_video_key(f), _frame_num(f)))
video_groups = defaultdict(list)
for f in all_fnames_sorted:
    video_groups[_video_key(f)].append(f)
video_groups = dict(video_groups)
print(f'\nVideo groups: {len(video_groups)} videos')

sweep_frames = []
for vk, frames in sorted(video_groups.items()):
    n = len(frames)
    for i in [n//4, n//2, 3*n//4]:
        sweep_frames.append(frames[i])
print(f'Sweep set   : {len(sweep_frames)} frames  (3 per video x {len(video_groups)} videos)')

cache_objects = {
    'gt_lookup'   : gt_lookup,    'dot_lookup'  : dot_lookup,
    'cal_frames'  : cal_frames,   'test_frames' : test_frames,
    'sweep_frames': sweep_frames, 'video_groups': video_groups,
    'gt_df'       : gt_df,        'tree_summary': tree_summary,
    'cohorts'     : COHORTS,
}
for name, obj in cache_objects.items():
    path = os.path.join(CACHE_DIR, f'{name}.pkl')
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)
    print(f'  Saved {name}.pkl')

print('\nDone. Load in any model notebook with:')
print('  from shared import load_cache')
print('  data = load_cache()')
