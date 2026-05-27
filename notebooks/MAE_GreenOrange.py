# %% [markdown]
# # MAE — Green & Unripe Orange Detection
# **Paper:** Masked Autoencoders Are Scalable Vision Learners — He et al., CVPR 2022
# **arXiv:** https://arxiv.org/abs/2111.06377
# **Institution:** Meta AI / FAIR
# **Approach:** ViT-B/16 self-supervised via masked image modelling — 75% patches masked, model reconstructs from visible context

# %% [markdown]
# ## 1. Configuration

# %%
import os
from config import TREE_ID_MAP, TREE_40SEC_RAW_IDS, CAL_RAW_IDS, TEST_RAW_IDS, RIPE_VIDEO_CONFIGS

BASE_DIR     = '/home/jovyan/OrangeGrove'
FRAMES_DIR   = os.path.join(BASE_DIR, 'frames')
DIR_30SEC    = os.path.join(FRAMES_DIR, '30sec')
DIR_40SEC    = os.path.join(FRAMES_DIR, '40sec')
WEIGHTS_DIR  = os.path.join(BASE_DIR, 'weights')
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'mae_pretrain_vit_base.pth')
WEIGHTS_URL  = 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth'
WEIGHTS_URL2 = 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth'
OUT_DIR      = os.path.join(BASE_DIR, 'results', '03_MAE')
RIPE_BASE    = os.path.join(BASE_DIR, 'ripe_validation')
RIPE_FRAMES  = os.path.join(RIPE_BASE, 'frames/40sec')
RIPE_ANNOT   = os.path.join(RIPE_BASE, 'annotations/40sec')
os.makedirs(OUT_DIR,     exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

MODEL_NAME    = 'MAE (ViT-B/16)'
MODEL_SLUG    = 'mae'
PRETRAINED_ON = 'ImageNet-1K'

IMG_SIZE   = 224
PATCH_SIZE = 16
GRID       = IMG_SIZE // PATCH_SIZE
EMBED_DIM  = 768

MAX_AREA         = 50_000_000
TREES_40SEC_SET  = set(TREE_ID_MAP[r] for r in TREE_40SEC_RAW_IDS)
TREES_30SEC      = [v for v in TREE_ID_MAP.values() if v not in TREES_40SEC_SET]
TREES_40SEC      = [TREE_ID_MAP[r] for r in TREE_40SEC_RAW_IDS]
VIDEOS           = ['Vid 01', 'Vid 02', 'Vid 03']
TREE_FOLDER_MAP  = TREE_ID_MAP
CAL_TREES        = [TREE_ID_MAP[r] for r in CAL_RAW_IDS]
TEST_TREES       = [TREE_ID_MAP[r] for r in TEST_RAW_IDS]
CAL_TREE_IDS     = CAL_RAW_IDS
TEST_TREE_IDS    = TEST_RAW_IDS

CONFIRMED_COUNTS = {
    ('Tree_01','Vid 01'):301,('Tree_01','Vid 02'):305,('Tree_01','Vid 03'):300,
    ('Tree_02','Vid 01'):300,('Tree_02','Vid 02'):300,('Tree_02','Vid 03'):300,
    ('Tree_03','Vid 01'):300,('Tree_03','Vid 02'):306,('Tree_03','Vid 03'):300,
    ('Tree_04','Vid 01'):300,('Tree_04','Vid 02'):315,('Tree_04','Vid 03'):316,
    ('Tree_05','Vid 01'):305,('Tree_05','Vid 02'):305,('Tree_05','Vid 03'):300,
    ('Tree_06','Vid 01'):400,('Tree_06','Vid 02'):403,('Tree_06','Vid 03'):400,
    ('Tree_07','Vid 01'):403,('Tree_07','Vid 02'):405,('Tree_07','Vid 03'):400,
    ('Tree_08','Vid 01'):404,('Tree_08','Vid 02'):400,('Tree_08','Vid 03'):405,
    ('Tree_09','Vid 01'):400,('Tree_09','Vid 02'):400,('Tree_09','Vid 03'):404,
    ('Tree_10','Vid 01'):400,('Tree_10','Vid 02'):400,('Tree_10','Vid 03'):400,
}
TOTAL_FRAMES = sum(CONFIRMED_COUNTS.values())
CAL_FRAMES   = sum(v for (t,_),v in CONFIRMED_COUNTS.items() if t in CAL_TREES)
TEST_FRAMES  = sum(v for (t,_),v in CONFIRMED_COUNTS.items() if t in TEST_TREES)

print(f'Model        : {MODEL_NAME}')
print(f'Total frames : {TOTAL_FRAMES:,}  |  Cal: {CAL_FRAMES:,}  |  Test: {TEST_FRAMES:,}')
print(f'Architecture : ViT-B/16 (timm)  patch={PATCH_SIZE}  grid={GRID}x{GRID}  embed={EMBED_DIM}')
print(f'Inference    : Full-image 4K→224px (no tiling)')

# %% [markdown]
# ## 2. Dependencies

# %%
import sys, subprocess, torch

print(f'Python  : {sys.version}')
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
    print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

subprocess.run([sys.executable,'-m','pip','install','numpy<2.0','--force-reinstall','-q'], capture_output=True)

for pkg in ['opencv-python','Pillow','pandas','openpyxl','matplotlib',
            'scikit-learn','scipy','tqdm','tabulate','torchvision','timm']:
    r = subprocess.run([sys.executable,'-m','pip','install',pkg,'-q'], capture_output=True)
    print(f"  {'OK' if r.returncode==0 else 'FAIL'} {pkg}")

import pandas as pd
import numpy as np
import cv2, time, json, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tabulate        import tabulate
from tqdm            import tqdm
from datetime        import timedelta
from PIL             import Image as PILImage
from sklearn.cluster import KMeans
from scipy           import ndimage
import torchvision.transforms as T
import timm

print(f'NumPy  : {np.__version__}  |  cv2: {cv2.__version__}  |  timm: {timm.__version__}')

# %% [markdown]
# ## 3. Download MAE Weights

# %%
WEIGHTS_LOADED_AS = None

if os.path.exists(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) > 100e6:
    print(f'Weights exist ({os.path.getsize(WEIGHTS_PATH)/1e6:.1f} MB)')
    WEIGHTS_LOADED_AS = 'cached'
else:
    for url, tag in [(WEIGHTS_URL,'mae_official'),(WEIGHTS_URL2,'mae_alternate')]:
        print(f'Downloading from {url[:65]}...')
        r = subprocess.run(['wget','-q','--show-progress','--timeout=120','-O',WEIGHTS_PATH,url],
                           capture_output=False)
        if r.returncode==0 and os.path.exists(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH)>100e6:
            print(f'Downloaded ({os.path.getsize(WEIGHTS_PATH)/1e6:.1f} MB)')
            WEIGHTS_LOADED_AS = tag; break
        if os.path.exists(WEIGHTS_PATH): os.remove(WEIGHTS_PATH)
    if not WEIGHTS_LOADED_AS:
        print('Both sources failed — upload weights manually to:')
        print(f'  {WEIGHTS_PATH}')
        WEIGHTS_LOADED_AS = 'MISSING'

if os.path.exists(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) > 100e6:
    ckpt = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=False)
    keys = list(ckpt.keys()) if isinstance(ckpt, dict) else ['raw_state_dict']
    print(f'Checkpoint keys: {keys}')

print(f'Weights source : {WEIGHTS_LOADED_AS}')

# %% [markdown]
# ## 4. Load MAE Model

# %%
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
mae_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)

state = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=False)
if   'model'      in state: weights = state['model']
elif 'state_dict' in state: weights = state['state_dict']
else:                        weights = state

missing, unexpected = mae_model.load_state_dict(weights, strict=False)
mae_model.eval().to(DEVICE)

decoder_missing = [k for k in missing if 'decoder' in k or 'mask' in k]
encoder_missing = [k for k in missing if 'decoder' not in k and 'mask' not in k]
print(f'Missing keys — decoder: {len(decoder_missing)} (expected)  encoder: {len(encoder_missing)} (should be 0)')
if encoder_missing: print(f'Unexpected encoder gaps: {encoder_missing[:5]}')

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def extract_mae_features(img_pil):
    tensor   = transform(img_pil).unsqueeze(0).to(DEVICE)
    captured = {}
    def hook_fn(module, input, output): captured['tokens'] = output
    handle = mae_model.norm.register_forward_hook(hook_fn)
    with torch.no_grad(): _ = mae_model(tensor)
    handle.remove()
    return captured['tokens'][0, 1:, :].cpu().numpy()

test_feats = extract_mae_features(PILImage.fromarray(
    np.random.randint(0,255,(3840,2160,3),dtype=np.uint8)))
assert test_feats.shape == (196,768), f'Expected (196,768), got {test_feats.shape}'

print(f'Device       : {DEVICE}')
print(f'Embed dim    : {mae_model.embed_dim}')
print(f'Feature shape: {test_feats.shape}')
print(f'Weights      : {WEIGHTS_LOADED_AS}')

# %% [markdown]
# ## 5. Load Ground Truth

# %%
sys.path.insert(0, '/home/jovyan/OrangeGrove/notebooks')
from shared import (load_cache, mae, rmse, bias, within_n, compute_f1, tp_mae, box_dot_match)

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
gt_df['split'] = gt_df['image_filename'].apply(lambda f: 'cal' if f in cal_set else 'test')

gt_master = gt_df.copy()
gt_master['source_tree']  = gt_master['tree'].map(TREE_FOLDER_MAP)
gt_master['source_video'] = gt_master['image_filename'].str.extract(r'(Vid \d+)')
gt_master['source_group'] = gt_master['tree'].apply(
    lambda t: '40sec' if t in TREES_40SEC_SET else '30sec')

cal_sweep = [f for f in sweep_frames if any(f.startswith(t) for t in CAL_TREE_IDS)]

print(f'GT frames  : {len(gt_lookup):,}  |  Cal frames: {len(cal_frames):,}  |  Test frames: {len(test_frames):,}')
print(f'Cal sweep  : {len(cal_sweep)} frames')

# %% [markdown]
# ## 6. Dataset Inventory

# %%
print(tabulate(tree_summary, headers='keys', tablefmt='pretty', showindex=False))

inventory_rows = []
for sec, trees in [('30sec',TREES_30SEC),('40sec',TREES_40SEC)]:
    frames_dir = DIR_30SEC if sec=='30sec' else DIR_40SEC
    for tree in trees:
        split_tag = 'CAL' if tree in CAL_TREES else 'TEST'
        for vid in VIDEOS:
            vid_path  = os.path.join(frames_dir, tree, vid)
            confirmed = CONFIRMED_COUNTS.get((tree,vid), 0)
            disk = len([f for f in os.listdir(vid_path) if f.endswith('.jpg')]) \
                   if os.path.exists(vid_path) else 0
            match = 'OK' if disk==confirmed else f'MISMATCH {disk}v{confirmed}'
            inventory_rows.append({'Group':sec,'Tree':tree,'Video':vid,
                                   'Split':split_tag,'Frames':disk,
                                   'Expected':confirmed,'Match':match})
inventory_df = pd.DataFrame(inventory_rows)

# %% [markdown]
# ## 7. Inference Pipeline

# %%
def is_green_detection(img_bgr, box,
                        ripe_hue_min=15, ripe_hue_max=40,
                        ripe_sat_min=120, ripe_val_min=160,
                        ripe_reject_ratio=0.08,
                        green_hue_min=25, green_hue_max=85,
                        green_sat_min=40, green_ratio_thresh=0.25):
    x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
    H,W = img_bgr.shape[:2]
    x1,y1 = max(0,x1),max(0,y1); x2,y2 = min(W,x2),min(H,y2)
    if x2<=x1 or y2<=y1: return False
    crop = img_bgr[y1:y2,x1:x2]
    if crop.size==0: return False
    hsv  = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    hue,sat,val = hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    ripe = ((hue>=ripe_hue_min)&(hue<=ripe_hue_max)&(sat>=ripe_sat_min)&(val>=ripe_val_min))
    if ripe.sum()/hue.size >= ripe_reject_ratio: return False
    green = ((hue>=green_hue_min)&(hue<=green_hue_max)&(sat>=green_sat_min))
    return (green.sum()/hue.size) >= green_ratio_thresh

def score_cluster_mae(labels, cluster_id, expected_count=13):
    mask          = (labels==cluster_id).astype(np.uint8)
    labeled,n_obj = ndimage.label(mask)
    if n_obj==0: return -1,[]
    blobs  = [np.where(labeled==i) for i in range(1,n_obj+1)]
    sizes  = [np.sum(labeled==i)   for i in range(1,n_obj+1)]
    dom_pen= 1.0 if max(sizes)/(GRID*GRID)<0.40 else 0.1
    c_score= np.exp(-0.5*((n_obj-expected_count)/8)**2)
    f_score= min(n_obj/15.0,1.0)
    return c_score*dom_pen*f_score, blobs

def count_oranges_mae(img_bgr, n_clusters=6, min_area=200, max_area=MAX_AREA):
    H,W     = img_bgr.shape[:2]
    img_pil = PILImage.fromarray(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB))
    feats   = extract_mae_features(img_pil)
    km      = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels  = km.fit_predict(feats).reshape(GRID,GRID)
    scale_y,scale_x = H/GRID, W/GRID
    best_score,best_cluster,best_blobs = -1,-1,[]
    for c in range(n_clusters):
        score,blobs = score_cluster_mae(labels,c)
        if score>best_score: best_score,best_cluster,best_blobs = score,c,blobs
    all_boxes = []
    for region in best_blobs:
        y1=int(region[0].min()*scale_y); y2=int(region[0].max()*scale_y+scale_y)
        x1=int(region[1].min()*scale_x); x2=int(region[1].max()*scale_x+scale_x)
        x1,y1=max(0,x1),max(0,y1); x2,y2=min(W,x2),min(H,y2)
        if x2<=x1 or y2<=y1: continue
        if (x2-x1)*(y2-y1) >= min_area: all_boxes.append([x1,y1,x2,y2])
    green_boxes = [b for b in all_boxes if is_green_detection(img_bgr,b)]
    return len(green_boxes), all_boxes, green_boxes, labels, best_cluster

print('Full-image MAE detection ready')
print(f'  Feature dim      : {EMBED_DIM} (vs iBOT 384)')
print(f'  Dominance thresh : 0.40 (vs iBOT 0.50)')
print(f'  Each patch covers ~{int(3840/GRID)}x{int(2160/GRID)}px in 4K frame')

# %% [markdown]
# ## 8. Parameter Calibration

# %%
N_CLUSTERS_RANGE = [4, 5, 6, 7, 8, 9, 10]
best_mae_val     = float('inf')
BEST_N_CLUSTERS  = 6
BEST_MIN_AREA    = 200
sweep_rows       = []

for n_clust in N_CLUSTERS_RANGE:
    preds,gts,skipped = [],[],0
    t_start = time.time()
    for fname in cal_sweep:
        gt_c = gt_lookup.get(fname)
        if gt_c is None: skipped+=1; continue
        try:
            tree_id = fname.split('_Vid')[0]
            vid     = f'Vid {fname.split("Vid ")[1].split("_")[0]}'
        except: skipped+=1; continue
        tree_folder = TREE_FOLDER_MAP.get(tree_id)
        if not tree_folder: skipped+=1; continue
        sec        = '40sec' if tree_id in TREE_40SEC_RAW_IDS else '30sec'
        frames_dir = DIR_40SEC if sec=='40sec' else DIR_30SEC
        img_path   = os.path.join(frames_dir, tree_folder, vid, fname)
        if not os.path.exists(img_path): skipped+=1; continue
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: skipped+=1; continue
        count,_,_,_,_ = count_oranges_mae(img_bgr, n_clusters=n_clust,
                                           min_area=200, max_area=MAX_AREA)
        preds.append(count); gts.append(int(gt_c))
    if len(preds)<5: continue
    arr_p,arr_g = np.array(preds,dtype=float), np.array(gts,dtype=float)
    m_mae  = float(np.mean(np.abs(arr_p-arr_g)))
    m_rmse = float(np.sqrt(np.mean((arr_p-arr_g)**2)))
    m_w2   = float(np.mean(np.abs(arr_p-arr_g)<=2)*100)
    m_bias = float(np.mean(arr_p-arr_g))
    is_best = m_mae < best_mae_val
    if is_best: best_mae_val,BEST_N_CLUSTERS = m_mae,n_clust
    print(f'  n_clusters={n_clust}  MAE={m_mae:.2f}  RMSE={m_rmse:.2f}  '
          f'W±2={m_w2:.1f}%  Bias={m_bias:+.2f}  ({time.time()-t_start:.0f}s)'
          f'{"  <- best" if is_best else ""}')
    sweep_rows.append({'n_clusters':n_clust,'MAE':round(m_mae,2),'RMSE':round(m_rmse,2),
                       'W±2%':round(m_w2,1),'Bias':round(m_bias,2),
                       'Frames':len(preds),'Skipped':skipped})

sweep_df = pd.DataFrame(sweep_rows)
print(f'\nSelected n_clusters : {BEST_N_CLUSTERS}  (cal MAE={best_mae_val:.2f})')

# %% [markdown]
# ## 9. Full Inference — 10,577 Frames

# %%
all_results    = []
skipped        = 0
total_filtered = 0
global_start   = time.time()

for sec, trees in [('30sec',TREES_30SEC),('40sec',TREES_40SEC)]:
    frames_dir = DIR_30SEC if sec=='30sec' else DIR_40SEC
    for tree in trees:
        split_tag = 'cal' if tree in CAL_TREES else 'test'
        for vid in VIDEOS:
            vid_path  = os.path.join(frames_dir, tree, vid)
            confirmed = CONFIRMED_COUNTS.get((tree,vid), 0)
            vid_rows  = gt_master[
                (gt_master['source_tree']==tree) &
                (gt_master['source_video']==vid)
            ].copy().reset_index(drop=True)
            if len(vid_rows)==0: continue
            vid_results=[]; vid_filtered=0
            for _, row in tqdm(vid_rows.iterrows(), total=len(vid_rows),
                               desc=f'[{split_tag.upper()}] {tree} {vid}',
                               unit='fr', bar_format='{l_bar}{bar:20}{r_bar}'):
                fname    = str(row['image_filename']).strip()
                img_path = os.path.join(vid_path, fname)
                if not os.path.exists(img_path): skipped+=1; continue
                img_bgr = cv2.imread(img_path)
                if img_bgr is None: skipped+=1; continue
                count,all_boxes,green_boxes,_,_ = count_oranges_mae(
                    img_bgr, n_clusters=BEST_N_CLUSTERS,
                    min_area=BEST_MIN_AREA, max_area=MAX_AREA)
                total_nms=len(all_boxes); filtered=total_nms-len(green_boxes)
                gt_c=int(row['ground_truth_count'])
                vid_filtered+=filtered; total_filtered+=filtered
                vid_results.append({
                    'group':sec,'tree':tree,'video':vid,'split':split_tag,
                    'image_filename':fname,'ground_truth':gt_c,'predicted':count,
                    'detections_pre_filter':total_nms,'filtered_out':filtered,
                    'error':count-gt_c,'abs_error':abs(count-gt_c),
                })
            all_results.extend(vid_results)

res_df = pd.DataFrame(all_results)
total_elapsed = time.time()-global_start
print(f'Frames processed : {len(res_df):,}  |  Skipped: {skipped}  |  Ripe filtered: {total_filtered:,}')
print(f'Total time       : {str(timedelta(seconds=int(total_elapsed)))}')

# %% [markdown]
# ## 10. Evaluation

# %%
%matplotlib inline
import matplotlib; matplotlib.rcParams['figure.dpi'] = 100

def compute_metrics(df):
    g,p = df['ground_truth'].values, df['predicted'].values
    e,ae = p-g, np.abs(p-g)
    return {'Frames':len(df),'MAE':round(float(np.mean(ae)),2),
            'RMSE':round(float(np.sqrt(np.mean(e**2))),2),
            'W+-2%':round(float(np.mean(ae<=2)*100),1),
            'Bias':round(float(np.mean(e)),2),
            'GT mean':round(float(np.mean(g)),2),
            'Pred mean':round(float(np.mean(p)),2),
            'Overcounts':int(np.sum(e>0)),'Undercounts':int(np.sum(e<0)),'Exact':int(np.sum(e==0))}

summary_table = pd.DataFrame([
    {'Split':'FULL (10 trees)',       **compute_metrics(res_df)},
    {'Split':'CAL  (6 trees)',        **compute_metrics(res_df[res_df['split']=='cal'])},
    {'Split':'TEST (4 trees) — MAIN', **compute_metrics(res_df[res_df['split']=='test'])},
])
print(tabulate(summary_table, headers='keys', tablefmt='pretty', showindex=False))

video_rows = []
for (grp,tree,vid,split), df_v in res_df.groupby(['group','tree','video','split']):
    video_rows.append({'Group':grp,'Tree':tree,'Video':vid,'Split':split,**compute_metrics(df_v)})
video_df = pd.DataFrame(video_rows)
print(tabulate(video_df[['Group','Tree','Video','Split','Frames','MAE','RMSE','W+-2%','Bias']],
               headers='keys', tablefmt='pretty', showindex=False))

tree_rows = []
for (grp,tree,split), df_t in res_df.groupby(['group','tree','split']):
    tree_rows.append({'Group':grp,'Tree':tree,'Split':split,**compute_metrics(df_t)})
tree_res_df = pd.DataFrame(tree_rows)
print(tabulate(tree_res_df[['Group','Tree','Split','Frames','MAE','RMSE','W+-2%','Bias']],
               headers='keys', tablefmt='pretty', showindex=False))

group_rows = []
for grp, df_g in res_df.groupby('group'):
    group_rows.append({'Group':grp,**compute_metrics(df_g)})
group_df = pd.DataFrame(group_rows)
print(tabulate(group_df, headers='keys', tablefmt='pretty', showindex=False))

spatial_results=[]; frames_no_dots=0

for sec, trees in [('30sec',TREES_30SEC),('40sec',TREES_40SEC)]:
    frames_dir = DIR_30SEC if sec=='30sec' else DIR_40SEC
    for tree in trees:
        split_tag = 'cal' if tree in CAL_TREES else 'test'
        for vid in VIDEOS:
            vid_path = os.path.join(frames_dir, tree, vid)
            vid_rows = gt_master[
                (gt_master['source_tree']==tree) &
                (gt_master['source_video']==vid)
            ].copy().reset_index(drop=True)
            if len(vid_rows)==0: continue
            for _, row in tqdm(vid_rows.iterrows(), total=len(vid_rows),
                               desc=f'[{split_tag.upper()}] {tree} {vid}',
                               unit='fr', bar_format='{l_bar}{bar:20}{r_bar}'):
                fname    = str(row['image_filename']).strip()
                img_path = os.path.join(vid_path, fname)
                if not os.path.exists(img_path): continue
                img_bgr = cv2.imread(img_path)
                if img_bgr is None: continue
                gt_c = int(row['ground_truth_count'])
                dots = dot_lookup.get(fname,[])
                if not dots: frames_no_dots+=1
                count,all_boxes,green_boxes,_,_ = count_oranges_mae(
                    img_bgr, n_clusters=BEST_N_CLUSTERS,
                    min_area=BEST_MIN_AREA, max_area=MAX_AREA)
                if not green_boxes:
                    spatial_results.append({'group':sec,'tree':tree,'video':vid,
                        'split':split_tag,'fname':fname,'gt_count':gt_c,'pred_count':0,
                        'tp':0,'fp':0,'fn':len(dots),'precision':0.0,'recall':0.0,'f1':0.0})
                    continue
                boxes_px = [(float(b[0]),float(b[1]),float(b[2]),float(b[3])) for b in green_boxes]
                tp,fp,fn = box_dot_match(boxes_px,dots)
                prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
                rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
                f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
                spatial_results.append({'group':sec,'tree':tree,'video':vid,
                    'split':split_tag,'fname':fname,'gt_count':gt_c,'pred_count':count,
                    'tp':tp,'fp':fp,'fn':fn,
                    'precision':round(prec,4),'recall':round(rec,4),'f1':round(f1,4)})

spatial_df = pd.DataFrame(spatial_results)

def spatial_metrics(df):
    ttp,tfp,tfn = df['tp'].sum(),df['fp'].sum(),df['fn'].sum()
    prec = ttp/(ttp+tfp) if (ttp+tfp)>0 else 0
    rec  = ttp/(ttp+tfn) if (ttp+tfn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    return {'Frames':len(df),'Total TP':int(ttp),'Total FP':int(tfp),'Total FN':int(tfn),
            'Precision':round(prec,4),'Recall':round(rec,4),'F1':round(f1,4),
            'TP-MAE':round(float(np.mean(np.abs(df['tp']-df['gt_count']))),2)}

overall_spatial = spatial_metrics(spatial_df)
cal_spatial     = spatial_metrics(spatial_df[spatial_df['split']=='cal'])
test_spatial    = spatial_metrics(spatial_df[spatial_df['split']=='test'])
spatial_summary = pd.DataFrame([
    {'Split':'FULL (10 trees)',       **overall_spatial},
    {'Split':'CAL  (6 trees)',        **cal_spatial},
    {'Split':'TEST (4 trees) — MAIN', **test_spatial},
])
print(tabulate(spatial_summary, headers='keys', tablefmt='pretty', showindex=False))

tree_spatial_rows = []
for (grp,tree,split), df_t in spatial_df.groupby(['group','tree','split']):
    tree_spatial_rows.append({'Group':grp,'Tree':tree,'Split':split,**spatial_metrics(df_t)})
tree_spatial_df = pd.DataFrame(tree_spatial_rows)
print(tabulate(tree_spatial_df[['Group','Tree','Split','Precision','Recall','F1','TP-MAE']],
               headers='keys', tablefmt='pretty', showindex=False))
print(f'Frames with no dot annotations: {frames_no_dots}')

gt   = res_df['ground_truth'].values
pred = res_df['predicted'].values
err  = pred-gt
mae_val,bias_val,w2_val = np.mean(np.abs(err)),np.mean(err),np.mean(np.abs(err)<=2)*100

fig, axes = plt.subplots(2,3, figsize=(20,12))
fig.suptitle(f'{MODEL_NAME}\n{len(res_df):,} frames  MAE={mae_val:.2f}  '
             f'W+-2={w2_val:.1f}%  Bias={bias_val:+.2f}', fontsize=13, fontweight='bold')

ax = axes[0,0]
colors = res_df['split'].map({'cal':'steelblue','test':'coral'})
ax.scatter(gt, pred, alpha=0.3, c=colors, s=8)
mn,mx = min(gt.min(),pred.min()), max(gt.max(),pred.max())
ax.plot([mn,mx],[mn,mx],'r--',lw=1.5)
ax.set_xlabel('GT'); ax.set_ylabel('Predicted'); ax.set_title(f'GT vs Predicted  MAE={mae_val:.2f}')
ax.legend(handles=[mpatches.Patch(color='steelblue',label='Calibration'),
                   mpatches.Patch(color='coral',label='Test')], fontsize=8)

ax = axes[0,1]
ax.hist(err, bins=40, color='saddlebrown', edgecolor='black', alpha=0.8)
ax.axvline(0, color='black',lw=1.5,linestyle='--')
ax.axvline(bias_val, color='red',lw=1.5,label=f'Bias={bias_val:+.2f}')
ax.set_xlabel('Error (pred-GT)'); ax.set_ylabel('Frames'); ax.set_title('Error Distribution')
ax.legend(fontsize=8)

ax = axes[0,2]
colors_t = ['#3498DB' if r['Split']=='cal' else '#E74C3C' for _,r in tree_res_df.iterrows()]
ax.bar(tree_res_df['Tree'], tree_res_df['MAE'], color=colors_t, edgecolor='black')
ax.axhline(mae_val, color='red',linestyle='--',lw=1.5,label=f'Overall MAE={mae_val:.2f}')
ax.set_xlabel('Tree'); ax.set_ylabel('MAE'); ax.set_title('MAE per Tree')
ax.tick_params(axis='x',rotation=45)
ax.legend(handles=[mpatches.Patch(color='#3498DB',label='Calibration'),
                   mpatches.Patch(color='#E74C3C',label='Test'),
                   plt.Line2D([0],[0],color='red',linestyle='--',label='Overall MAE')], fontsize=8)

ax = axes[1,0]
pivot = video_df.pivot_table(index='Tree',columns='Video',values='MAE',aggfunc='mean')
im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=8)
ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index, fontsize=8)
ax.set_title('MAE Heatmap (green=low, red=high)')
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.values[i,j]
        if not np.isnan(v): ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=7)
plt.colorbar(im, ax=ax)

colors_t2 = ['#3498DB' if r['Split']=='cal' else '#E74C3C' for _,r in tree_spatial_df.iterrows()]
ax = axes[1,1]
ax.scatter(tree_spatial_df['Recall'], tree_spatial_df['Precision'],
           c=colors_t2, s=120, edgecolors='black', zorder=5)
for _, r in tree_spatial_df.iterrows():
    ax.annotate(r['Tree'],(r['Recall'],r['Precision']),
                textcoords='offset points',xytext=(6,4),fontsize=7)
ax.axhline(overall_spatial['Precision'],color='red',linestyle='--',lw=1.5)
ax.axvline(overall_spatial['Recall'],color='blue',linestyle='--',lw=1.5)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision vs Recall per Tree')
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(True,alpha=0.3)
ax.legend(handles=[mpatches.Patch(color='#3498DB',label='CAL'),
                   mpatches.Patch(color='#E74C3C',label='TEST'),
                   plt.Line2D([0],[0],color='red',linestyle='--',
                              label=f'P={overall_spatial["Precision"]:.3f}'),
                   plt.Line2D([0],[0],color='blue',linestyle='--',
                              label=f'R={overall_spatial["Recall"]:.3f}')], fontsize=8)

ax = axes[1,2]
ax.bar(tree_spatial_df['Tree'], tree_spatial_df['F1'], color=colors_t2, edgecolor='black')
ax.axhline(overall_spatial['F1'],color='red',linestyle='--',lw=1.5,
           label=f'Overall F1={overall_spatial["F1"]:.3f}')
ax.set_xlabel('Tree'); ax.set_ylabel('F1'); ax.set_title('F1 per Tree')
ax.tick_params(axis='x',rotation=45); ax.set_ylim(0,1)
ax.legend(handles=[mpatches.Patch(color='#3498DB',label='CAL'),
                   mpatches.Patch(color='#E74C3C',label='TEST'),
                   plt.Line2D([0],[0],color='red',linestyle='--',
                              label=f'F1={overall_spatial["F1"]:.3f}')], fontsize=8)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, 'mae_evaluation.png')
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved: {plot_path}')

# %% [markdown]
# ## 11. Ripe Orange Suppression Validation

# %%
RIPE_VIDEOS      = RIPE_VIDEO_CONFIGS
all_ripe_results = []
video_summary    = []

for vc in RIPE_VIDEOS:
    tree = vc['tree']; vid = vc['vid']
    json_path  = os.path.join(RIPE_ANNOT, vc['annot_dir'], vc['json'])
    frames_dir = os.path.join(RIPE_FRAMES, tree, vid)
    with open(json_path) as f: data = json.load(f)
    frame_names = data['frameNames']; annotations = data['annotations']
    vid_total=0; vid_supp=0; vid_fail=0
    for idx_str, ripe_pts in tqdm(annotations.items(), desc=f'  {tree}/{vid}', unit='frame'):
        idx      = int(idx_str)
        fname    = frame_names[idx]
        img_path = os.path.join(frames_dir, fname)
        if not os.path.exists(img_path): continue
        img_bgr = cv2.imread(img_path)
        _,raw_boxes,green_boxes,_,_ = count_oranges_mae(
            img_bgr, n_clusters=BEST_N_CLUSTERS,
            min_area=BEST_MIN_AREA, max_area=MAX_AREA)
        for p in ripe_pts:
            px,py    = p['x'],p['y']
            in_raw   = any(b[0]<=px<=b[2] and b[1]<=py<=b[3] for b in raw_boxes)
            in_green = any(b[0]<=px<=b[2] and b[1]<=py<=b[3] for b in green_boxes)
            if in_raw and in_green: vid_fail+=1
            else:                   vid_supp+=1
            vid_total+=1
        all_ripe_results.append({'tree':tree,'video':vid,'frame':fname,'ripe_pts':len(ripe_pts)})
    rate = vid_supp/vid_total*100 if vid_total>0 else 0
    print(f'  {tree}/{vid}: {vid_total} pts  suppressed={vid_supp}  failed={vid_fail}  rate={rate:.1f}%')
    video_summary.append({'Tree':tree,'Video':vid,'Total':vid_total,
                          'Suppressed':vid_supp,'Failed':vid_fail,'Rate%':round(rate,1)})

vsum_df      = pd.DataFrame(video_summary)
total_ripe   = vsum_df['Total'].sum()
total_supp   = vsum_df['Suppressed'].sum()
overall_rate = total_supp/total_ripe*100 if total_ripe>0 else 0
print(tabulate(vsum_df, headers='keys', tablefmt='pretty', showindex=False))
print(f'Total ripe points : {total_ripe}  |  Suppression rate: {overall_rate:.1f}%')

# %% [markdown]
# ## 12. Export Results

# %%
out_path = os.path.join(OUT_DIR, 'mae_results.xlsx')
test_m   = compute_metrics(res_df[res_df['split']=='test'])
full_m   = compute_metrics(res_df)

summary = pd.DataFrame([{
    'Model':MODEL_NAME,'Pretrained On':PRETRAINED_ON,
    'Weights Source':WEIGHTS_LOADED_AS,'Architecture':'ViT-B/16 (timm)',
    'Embed Dim':EMBED_DIM,'n_clusters':BEST_N_CLUSTERS,
    'min_area':BEST_MIN_AREA,'max_area':MAX_AREA,
    'Inference':'Full-image (no tiling)','Total Frames':len(res_df),
    'Test MAE':test_m['MAE'],'Test RMSE':test_m['RMSE'],
    'Test W+-2%':test_m['W+-2%'],'Test Bias':test_m['Bias'],
    'Full MAE':full_m['MAE'],'Full RMSE':full_m['RMSE'],
    'Full W+-2%':full_m['W+-2%'],'Full Bias':full_m['Bias'],
    'Ripe Filtered':total_filtered,'Ripe Suppression%':round(overall_rate,1),
}])

with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    summary.to_excel(        writer, sheet_name='Summary',          index=False)
    res_df.to_excel(         writer, sheet_name='Frame Results',    index=False)
    video_df.to_excel(       writer, sheet_name='Per-Video',        index=False)
    tree_res_df.to_excel(    writer, sheet_name='Per-Tree',         index=False)
    group_df.to_excel(       writer, sheet_name='Group Comparison', index=False)
    summary_table.to_excel(  writer, sheet_name='Cal vs Test',      index=False)
    sweep_df.to_excel(       writer, sheet_name='Parameter Sweep',  index=False)
    vsum_df.to_excel(        writer, sheet_name='Ripe Validation',  index=False)
    spatial_df.to_excel(     writer, sheet_name='Spatial Metrics',  index=False)
    tree_spatial_df.to_excel(writer, sheet_name='Spatial Per-Tree', index=False)
    inventory_df.to_excel(   writer, sheet_name='Inventory',        index=False)

print(f'Saved: {out_path}')
print(f'TEST  — MAE: {test_m["MAE"]}  RMSE: {test_m["RMSE"]}  W+-2%: {test_m["W+-2%"]}  Bias: {test_m["Bias"]}')
print(f'FULL  — MAE: {full_m["MAE"]}  RMSE: {full_m["RMSE"]}  W+-2%: {full_m["W+-2%"]}  Bias: {full_m["Bias"]}')
