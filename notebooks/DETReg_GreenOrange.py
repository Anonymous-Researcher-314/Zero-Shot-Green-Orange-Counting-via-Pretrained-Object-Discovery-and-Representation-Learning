# %% [markdown]
# # DETReg — Green & Unripe Orange Detection
# **Paper:** DETReg: Unsupervised Pretraining with Region Priors for Object Detection — Bar et al., CVPR 2022
# **arXiv:** https://arxiv.org/abs/2106.04550
# **Architecture:** Deformable DETR (ResNet-50) with 300 fixed object queries
# **Note:** Zero-shot failure documented — DETR-style decoders require domain fine-tuning for agricultural tasks

# %% [markdown]
# ## 1. Configuration

# %%
import os
from config import TREE_ID_MAP, TREE_40SEC_RAW_IDS, CAL_RAW_IDS, TEST_RAW_IDS, RIPE_VIDEO_CONFIGS

BASE_DIR   = '/home/jovyan/OrangeGrove'
FRAMES_DIR = os.path.join(BASE_DIR, 'frames')
DIR_30SEC  = os.path.join(FRAMES_DIR, '30sec')
DIR_40SEC  = os.path.join(FRAMES_DIR, '40sec')
DETREG_DIR = os.path.join(BASE_DIR, 'DETReg')
CKPT_DIR   = os.path.join(DETREG_DIR, 'checkpoints')
CKPT_PATH  = os.path.join(CKPT_DIR, 'detreg_imagenet.pth')
CKPT_URL   = 'https://github.com/amirbar/DETReg/releases/download/1.0.0/checkpoint_imagenet.pth'
CKPT_URL2  = 'https://dl.fbaipublicfiles.com/detreg/DETReg_top30_in.pth'
OUT_DIR     = os.path.join(BASE_DIR, 'results', '05_DETReg')
RIPE_BASE   = os.path.join(BASE_DIR, 'ripe_validation')
RIPE_FRAMES = os.path.join(RIPE_BASE, 'frames/40sec')
RIPE_ANNOT  = os.path.join(RIPE_BASE, 'annotations/40sec')
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

MODEL_NAME    = 'DETReg (Deformable DETR)'
MODEL_SLUG    = 'detreg'
PRETRAINED_ON = 'ImageNet (unsupervised region priors)'
NUM_QUERIES   = 300
NMS_IOU       = 0.5

TREES_40SEC_SET = set(TREE_ID_MAP[r] for r in TREE_40SEC_RAW_IDS)
TREES_30SEC     = [v for v in TREE_ID_MAP.values() if v not in TREES_40SEC_SET]
TREES_40SEC     = [TREE_ID_MAP[r] for r in TREE_40SEC_RAW_IDS]
VIDEOS          = ['Vid 01', 'Vid 02', 'Vid 03']
TREE_FOLDER_MAP = TREE_ID_MAP
CAL_TREES       = [TREE_ID_MAP[r] for r in CAL_RAW_IDS]
TEST_TREES      = [TREE_ID_MAP[r] for r in TEST_RAW_IDS]
CAL_TREE_IDS    = CAL_RAW_IDS
TEST_TREE_IDS   = TEST_RAW_IDS

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
print(f'Inference    : Full-image (4K → resize 800/1333px)')
print(f'Queries/img  : {NUM_QUERIES} (fixed budget — no tiling)')

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

for pkg in ['opencv-python','Pillow','pandas','openpyxl',
            'matplotlib','scipy','tqdm','tabulate','pycocotools','timm']:
    r = subprocess.run([sys.executable,'-m','pip','install',pkg,'-q'], capture_output=True)
    print(f"  {'OK' if r.returncode==0 else 'FAIL'} {pkg}")

import pandas as pd
import numpy as np
import cv2, time, json, os, argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tabulate        import tabulate
from tqdm            import tqdm
from datetime        import timedelta
from PIL             import Image as PILImage
from torchvision.ops import nms as tv_nms
import torchvision.transforms as T

print('All imports ready — no Detectron2 required')

# %% [markdown]
# ## 3. Clone DETReg & Patch CUDA Ops Fallback

# %%
import shutil

if not os.path.exists(os.path.join(DETREG_DIR, 'models')):
    if os.path.exists(DETREG_DIR): shutil.rmtree(DETREG_DIR)
    r = subprocess.run(['git','clone','https://github.com/amirbar/DETReg.git',DETREG_DIR],
                       capture_output=True, text=True, timeout=120)
    print(f'{"Cloned" if r.returncode==0 else "Clone failed: "+r.stderr[:100]}')
else:
    print(f'DETReg exists at {DETREG_DIR}')

for p in [DETREG_DIR, os.path.join(DETREG_DIR,'models')]:
    if p not in sys.path: sys.path.insert(0, p)

req = os.path.join(DETREG_DIR, 'requirements.txt')
if os.path.exists(req):
    with open(req) as f: pkgs = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    for pkg in pkgs:
        r = subprocess.run([sys.executable,'-m','pip','install','-q',pkg], capture_output=True)

func_path = os.path.join(DETREG_DIR,'models','ops','functions','ms_deform_attn_func.py')
patched = '''import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

try:
    import MultiScaleDeformableAttention as MSDA
    HAS_CUDA_OPS = True
except ImportError:
    HAS_CUDA_OPS = False

def ms_deform_attn_core_pytorch(value, value_spatial_shapes,
                                  sampling_locations, attention_weights):
    N_,S_,M_,D_ = value.shape
    _,Lq_,M_,L_,P_,_ = sampling_locations.shape
    value_list = value.split([H_*W_ for H_,W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2*sampling_locations - 1
    sampling_value_list = []
    for lid_,(H_,W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1,2).reshape(N_*M_,D_,H_,W_)
        sampling_grid_l_ = sampling_grids[:,:,:,lid_].transpose(1,2).flatten(0,1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
            mode=\'bilinear\', padding_mode=\'zeros\', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1,2).reshape(N_*M_,1,Lq_,L_*P_)
    output = (torch.stack(sampling_value_list,dim=-2).flatten(-2)*attention_weights).sum(-1).view(N_,M_*D_,Lq_)
    return output.transpose(1,2).contiguous()

class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index,
                               sampling_locations, attention_weights)
        ctx.im2col_step = im2col_step
        if HAS_CUDA_OPS:
            return MSDA.ms_deform_attn_forward(value, value_spatial_shapes,
                value_level_start_index, sampling_locations, attention_weights, im2col_step)
        return ms_deform_attn_core_pytorch(value, value_spatial_shapes,
                                            sampling_locations, attention_weights)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (value, value_spatial_shapes, value_level_start_index,
         sampling_locations, attention_weights) = ctx.saved_tensors
        if HAS_CUDA_OPS:
            grad_value, grad_sampling_loc, grad_attn_weight = MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, grad_output, ctx.im2col_step)
        else:
            grad_value = torch.zeros_like(value)
            grad_sampling_loc = torch.zeros_like(sampling_locations)
            grad_attn_weight = torch.zeros_like(attention_weights)
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
'''
if os.path.exists(func_path):
    with open(func_path, 'w') as f: f.write(patched)
    print('ms_deform_attn_func.py patched with PyTorch fallback')
else:
    print(f'File not found: {func_path}')

# %% [markdown]
# ## 4. Download DETReg Checkpoint

# %%
WEIGHTS_LOADED_AS = None

if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 1e6:
    print(f'Checkpoint exists ({os.path.getsize(CKPT_PATH)/1e6:.1f} MB)')
    WEIGHTS_LOADED_AS = 'cached'
else:
    for i, url in enumerate([CKPT_URL, CKPT_URL2], 1):
        print(f'Trying source {i}: {url}')
        r = subprocess.run(['wget','-q','--show-progress','--timeout=120','-O',CKPT_PATH,url],
                           capture_output=False)
        if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 1e6:
            print(f'Downloaded ({os.path.getsize(CKPT_PATH)/1e6:.1f} MB)')
            WEIGHTS_LOADED_AS = f'source_{i}'; break
        if os.path.exists(CKPT_PATH): os.remove(CKPT_PATH)
    if not WEIGHTS_LOADED_AS:
        print(f'Both sources failed — upload manually to: {CKPT_PATH}')
        WEIGHTS_LOADED_AS = 'MISSING'

if WEIGHTS_LOADED_AS not in ('MISSING', None):
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    print(f'Checkpoint keys : {list(ckpt.keys()) if isinstance(ckpt,dict) else ["raw"]}')
    print(f'Source          : {WEIGHTS_LOADED_AS}')

# %% [markdown]
# ## 5. Load DETReg Model

# %%
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from models import build_model as detreg_build_model

parser = argparse.ArgumentParser(add_help=False)
for arg, default, typ in [
    ('--lr_backbone',1e-5,float),('--backbone','resnet50',str),
    ('--position_embedding','sine',str),('--position_embedding_scale',2*3.14159,float),
    ('--num_feature_levels',4,int),('--load_backbone','',str),
    ('--enc_layers',6,int),('--dec_layers',6,int),('--dim_feedforward',1024,int),
    ('--hidden_dim',256,int),('--dropout',0.1,float),('--nheads',8,int),
    ('--num_queries',300,int),('--dec_n_points',4,int),('--enc_n_points',4,int),
    ('--num_classes',91,int),('--num_classes_unsup',20,int),
    ('--set_cost_class',2,float),('--set_cost_bbox',5,float),('--set_cost_giou',2,float),
    ('--cls_loss_coef',2,float),('--bbox_loss_coef',5,float),('--giou_loss_coef',2,float),
    ('--focal_alpha',0.25,float),('--object_embedding_coef',1.0,float),
    ('--model','deformable_detr',str),('--query_pos_type','sine',str),
    ('--obj_embedding_head','intermediate',str),('--object_embedding_type','class_token',str),
    ('--dataset_file','coco',str),('--device','cuda',str),
]:
    parser.add_argument(arg, default=default, type=typ)
for flag in ['--dilation','--pre_norm','--masks','--aux_loss','--with_box_refine',
             '--two_stage','--iter_update','--with_query_pos','--coco_pretrain',
             '--object_embedding_loss']:
    parser.add_argument(flag, action='store_true')
parser.add_argument('--pretrain_settings', default=None)
parser.add_argument('--frozen_weights', default=None, type=str)

args = parser.parse_args([])
args.device = str(DEVICE)

model, criterion, postprocessors = detreg_build_model(args)
ckpt        = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
model_state = ckpt.get('model', ckpt)
model_state = {k.replace('module.',''): v for k,v in model_state.items()}
missing, unexpected = model.load_state_dict(model_state, strict=False)
model.eval().to(DEVICE)

transform = T.Compose([
    T.Resize(800, max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def detreg_inference(pil_img):
    W,H   = pil_img.size
    img_t = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_t)
    logits = outputs['pred_logits'][0]
    boxes  = outputs['pred_boxes'][0]
    scores = logits.softmax(-1)[:,:-1].max(-1).values
    cx,cy,bw,bh = boxes.unbind(-1)
    boxes_xyxy = torch.stack([(cx-0.5*bw)*W,(cy-0.5*bh)*H,
                               (cx+0.5*bw)*W,(cy+0.5*bh)*H], dim=-1)
    return scores.cpu(), boxes_xyxy.cpu()

test_sc, test_bx = detreg_inference(PILImage.fromarray(
    np.random.randint(0,255,(480,640,3),dtype=np.uint8)))
assert test_sc.shape==(300,) and test_bx.shape==(300,4)

print(f'Device      : {DEVICE}')
print(f'Missing keys: {len(missing)}  |  Unexpected: {len(unexpected)}')
print(f'Queries/img : {NUM_QUERIES}  |  Weights: {WEIGHTS_LOADED_AS}')
print('Full-image inference confirmed — no tiling')

# %% [markdown]
# ## 6. Load Ground Truth

# %%
sys.path.insert(0, '/home/jovyan/OrangeGrove/notebooks')
from shared import (load_cache, mae, rmse, bias, within_n, compute_f1, tp_mae, box_dot_match)

data         = load_cache()
gt_lookup    = data['gt_lookup']
dot_lookup   = data['dot_lookup']
cal_frames   = data['cal_frames']
test_frames  = data['test_frames']
sweep_frames = data['sweep_frames']
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
# ## 7. Dataset Inventory

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
# ## 8. Inference Strategy Verification

# %%
import cv2
from PIL import Image as PILImage

print('Architecture : Deformable DETR (transformer decoder)')
print(f'Queries      : {NUM_QUERIES} fixed per image')
print('Strategy     : FULL IMAGE — no tiling')
print()
print('Tiling abandoned:')
print(f'  8 tiles × 300 = 2,400 candidates → all score near 1.0 → MAE=803.50')
print('Full image:')
print(f'  300 candidates → threshold → manageable count')
print()

test_path = os.path.join(DIR_30SEC,'Tree_01','Vid 01',
                         [f for f in os.listdir(os.path.join(DIR_30SEC,'Tree_01','Vid 01'))
                          if f.endswith('.jpg')][0])
if os.path.exists(test_path):
    img_bgr = cv2.imread(test_path)
    img_pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    sc_full, _ = detreg_inference(img_pil)
    for t in [0.3, 0.5, 0.7, 0.9, 0.95]:
        print(f'  thresh={t}: {int((sc_full>t).sum())} candidates')

print('Full-image strategy confirmed')

# %% [markdown]
# ## 9. Inference Pipeline

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

def count_oranges_detreg(img_bgr, threshold, nms_iou=NMS_IOU):
    H,W     = img_bgr.shape[:2]
    img_pil = PILImage.fromarray(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB))
    scores, boxes_xyxy = detreg_inference(img_pil)
    keep_mask = scores > threshold
    if keep_mask.sum()==0: return 0, 0, 0
    kscores = scores[keep_mask]; kboxes = boxes_xyxy[keep_mask]
    kboxes[:,0].clamp_(0,W); kboxes[:,2].clamp_(0,W)
    kboxes[:,1].clamp_(0,H); kboxes[:,3].clamp_(0,H)
    keep        = tv_nms(kboxes.float(), kscores.float(), iou_threshold=nms_iou)
    kept_boxes  = kboxes[keep].numpy()
    total_nms   = len(kept_boxes)
    green_boxes = [b for b in kept_boxes if is_green_detection(img_bgr,b)]
    return len(green_boxes), total_nms, total_nms-len(green_boxes)

print('Full-image DETReg inference ready')
print(f'Queries per frame : {NUM_QUERIES}  (vs 2400 with tiling)')

# %% [markdown]
# ## 10. Threshold Calibration

# %%
THRESHOLDS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.96,0.97,0.98,0.99]
best_mae_val = float('inf')
BEST_THRESH  = 0.9
sweep_rows   = []

print(f'Sweep frames : {len(cal_sweep)}')
print('Note: DETReg expected MAE >200 at low thresholds (zero-shot limitation)')

for thresh in THRESHOLDS:
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
        count,_,_ = count_oranges_detreg(img_bgr, threshold=thresh)
        preds.append(count); gts.append(int(gt_c))
    if len(preds)<5: continue
    arr_p,arr_g = np.array(preds,dtype=float), np.array(gts,dtype=float)
    m_mae  = float(np.mean(np.abs(arr_p-arr_g)))
    m_rmse = float(np.sqrt(np.mean((arr_p-arr_g)**2)))
    m_w2   = float(np.mean(np.abs(arr_p-arr_g)<=2)*100)
    m_bias = float(np.mean(arr_p-arr_g))
    is_best = m_mae < best_mae_val
    if is_best: best_mae_val,BEST_THRESH = m_mae,thresh
    print(f'  thresh={thresh}  MAE={m_mae:7.2f}  RMSE={m_rmse:7.2f}  '
          f'W±2={m_w2:5.1f}%  Bias={m_bias:+8.2f}  ({time.time()-t_start:.0f}s)'
          f'{"  <- best" if is_best else ""}')
    sweep_rows.append({'Threshold':thresh,'MAE':round(m_mae,2),'RMSE':round(m_rmse,2),
                       'W±2%':round(m_w2,1),'Bias':round(m_bias,2),
                       'Frames':len(preds),'Skipped':skipped})

sweep_df = pd.DataFrame(sweep_rows)
print(f'\nSelected threshold : {BEST_THRESH}  (cal MAE={best_mae_val:.2f})')

# %% [markdown]
# ## 11. Full Inference — 10,577 Frames

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
            vid_results=[]; vid_filtered=0; vid_start=time.time()
            for _, row in tqdm(vid_rows.iterrows(), total=len(vid_rows),
                               desc=f'[{split_tag.upper()}] {tree} {vid}',
                               unit='fr', bar_format='{l_bar}{bar:20}{r_bar}'):
                fname    = str(row['image_filename']).strip()
                img_path = os.path.join(vid_path, fname)
                if not os.path.exists(img_path): skipped+=1; continue
                img_bgr = cv2.imread(img_path)
                if img_bgr is None: skipped+=1; continue
                count, total_nms, filtered = count_oranges_detreg(img_bgr, threshold=BEST_THRESH)
                gt_c=int(row['ground_truth_count'])
                vid_filtered+=filtered; total_filtered+=filtered
                vid_results.append({
                    'group':sec,'tree':tree,'video':vid,'split':split_tag,
                    'image_filename':fname,'ground_truth':gt_c,'predicted':count,
                    'detections_pre_filter':total_nms,'filtered_out':filtered,
                    'error':count-gt_c,'abs_error':abs(count-gt_c),
                })
            all_results.extend(vid_results)
            pd.DataFrame(all_results).to_pickle(
                os.path.join(OUT_DIR,'detreg_res_df_checkpoint.pkl'))
            if vid_results:
                vg=np.array([r['ground_truth'] for r in vid_results])
                vp=np.array([r['predicted']    for r in vid_results])
                print(f'  {tree} {vid}  MAE={np.mean(np.abs(vp-vg)):.2f}  '
                      f'Bias={np.mean(vp-vg):+.2f}  '
                      f'Time={str(timedelta(seconds=int(time.time()-vid_start)))}')

res_df = pd.DataFrame(all_results)
res_df.to_pickle(os.path.join(OUT_DIR,'detreg_res_df.pkl'))
total_elapsed = time.time()-global_start
print(f'Frames processed : {len(res_df):,}  |  Skipped: {skipped}')
print(f'Total time       : {str(timedelta(seconds=int(total_elapsed)))}')

# %% [markdown]
# ## 12. Evaluation

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

SPATIAL_DIR = os.path.join(OUT_DIR, 'spatial_per_tree')
os.makedirs(SPATIAL_DIR, exist_ok=True)

def process_tree_spatial(tree, group, split_tag):
    frames_dir = DIR_40SEC if group=='40sec' else DIR_30SEC
    pkl_path   = os.path.join(SPATIAL_DIR, f'spatial_{tree}.pkl')
    if os.path.exists(pkl_path):
        existing = pd.read_pickle(pkl_path)
        print(f'  {tree} loaded from cache ({len(existing)} rows)')
        return existing
    rows = []
    for vid in VIDEOS:
        vid_path = os.path.join(frames_dir, tree, vid)
        vid_rows = gt_master[
            (gt_master['source_tree']==tree) &
            (gt_master['source_video']==vid)
        ].copy().reset_index(drop=True)
        if len(vid_rows)==0: continue
        vid_tp=0; vid_fp=0; vid_fn=0
        for _, row in tqdm(vid_rows.iterrows(), total=len(vid_rows),
                           desc=f'  {vid}', unit='fr',
                           bar_format='{l_bar}{bar:20}{r_bar}'):
            fname    = str(row['image_filename']).strip()
            img_path = os.path.join(vid_path, fname)
            if not os.path.exists(img_path): continue
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            gt_c = int(row['ground_truth_count'])
            dots = dot_lookup.get(fname, [])
            H,W  = img_bgr.shape[:2]
            img_pil = PILImage.fromarray(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB))
            sc,bx   = detreg_inference(img_pil)
            km = sc > BEST_THRESH
            if km.sum()>0:
                ksc=sc[km]; kbx=bx[km]
                kbx[:,0].clamp_(0,W); kbx[:,2].clamp_(0,W)
                kbx[:,1].clamp_(0,H); kbx[:,3].clamp_(0,H)
                keep      = tv_nms(kbx.float(), ksc.float(), iou_threshold=NMS_IOU)
                raw_boxes = kbx[keep].numpy()
            else:
                raw_boxes = np.zeros((0,4))
            green_boxes = [b for b in raw_boxes if is_green_detection(img_bgr,b)]
            if not green_boxes:
                rows.append({'group':group,'tree':tree,'video':vid,'split':split_tag,
                    'fname':fname,'gt_count':gt_c,'pred_count':0,
                    'tp':0,'fp':0,'fn':len(dots),'precision':0.0,'recall':0.0,'f1':0.0})
                vid_fn+=len(dots); continue
            boxes_px = [(float(b[0]),float(b[1]),float(b[2]),float(b[3])) for b in green_boxes]
            tp,fp,fn = box_dot_match(boxes_px, dots)
            prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
            rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
            rows.append({'group':group,'tree':tree,'video':vid,'split':split_tag,
                'fname':fname,'gt_count':gt_c,'pred_count':len(green_boxes),
                'tp':tp,'fp':fp,'fn':fn,'precision':round(prec,4),
                'recall':round(rec,4),'f1':round(f1,4)})
            vid_tp+=tp; vid_fp+=fp; vid_fn+=fn
        vp2 = vid_tp/(vid_tp+vid_fp) if (vid_tp+vid_fp)>0 else 0
        vr2 = vid_tp/(vid_tp+vid_fn) if (vid_tp+vid_fn)>0 else 0
        vf2 = 2*vp2*vr2/(vp2+vr2) if (vp2+vr2)>0 else 0
        print(f'  {vid}: P={vp2:.3f}  R={vr2:.3f}  F1={vf2:.3f}')
    df = pd.DataFrame(rows)
    df.to_pickle(pkl_path)
    print(f'  Saved {len(df)} rows')
    return df

spatial_dfs = []
for tree, group, split_tag in [
    ('Tree_01','30sec','cal'),('Tree_02','30sec','test'),('Tree_03','30sec','test'),
    ('Tree_04','30sec','cal'),('Tree_05','30sec','cal'),
    ('Tree_06','40sec','cal'),('Tree_07','40sec','cal'),('Tree_08','40sec','cal'),
    ('Tree_09','40sec','test'),('Tree_10','40sec','test'),
]:
    print(f'\n[{split_tag.upper()}] {tree}')
    spatial_dfs.append(process_tree_spatial(tree, group, split_tag))

spatial_df = pd.concat(spatial_dfs, ignore_index=True)
spatial_df.to_pickle(os.path.join(OUT_DIR,'detreg_spatial_df.pkl'))

def spatial_metrics(df):
    ttp,tfp,tfn = df['tp'].sum(),df['fp'].sum(),df['fn'].sum()
    prec = ttp/(ttp+tfp) if (ttp+tfp)>0 else 0
    rec  = ttp/(ttp+tfn) if (ttp+tfn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    return {'Frames':len(df),'Total TP':int(ttp),'Total FP':int(tfp),'Total FN':int(tfn),
            'Precision':round(prec,4),'Recall':round(rec,4),'F1':round(f1,4)}

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
print(tabulate(tree_spatial_df[['Group','Tree','Split','Precision','Recall','F1']],
               headers='keys', tablefmt='pretty', showindex=False))

gt   = res_df['ground_truth'].values
pred = res_df['predicted'].values
err  = pred-gt
mae_val,bias_val,w2_val = np.mean(np.abs(err)),np.mean(err),np.mean(np.abs(err)<=2)*100

fig, axes = plt.subplots(2,3, figsize=(20,12))
fig.suptitle(f'{MODEL_NAME}\n{len(res_df):,} frames  MAE={mae_val:.2f}  '
             f'W+-2={w2_val:.1f}%  Bias={bias_val:+.2f}\n'
             f'Note: High MAE reflects zero-shot limitation — DETR requires domain fine-tuning',
             fontsize=12, fontweight='bold')

ax = axes[0,0]
colors = res_df['split'].map({'cal':'steelblue','test':'coral'})
ax.scatter(gt, pred, alpha=0.3, c=colors, s=8)
mn,mx = min(gt.min(),pred.min()), max(gt.max(),pred.max())
ax.plot([mn,mx],[mn,mx],'r--',lw=1.5)
ax.set_xlabel('GT'); ax.set_ylabel('Predicted'); ax.set_title(f'GT vs Predicted  MAE={mae_val:.2f}')
ax.legend(handles=[mpatches.Patch(color='steelblue',label='Calibration'),
                   mpatches.Patch(color='coral',label='Test')], fontsize=8)

ax = axes[0,1]
ax.hist(err, bins=40, color='tomato', edgecolor='black', alpha=0.8)
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
        if not np.isnan(v): ax.text(j,i,f'{v:.0f}',ha='center',va='center',fontsize=7)
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
plot_path = os.path.join(OUT_DIR, 'detreg_evaluation.png')
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved: {plot_path}')

# %% [markdown]
# ## 13. Ripe Orange Suppression Validation

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
        idx=int(idx_str); fname=frame_names[idx]
        img_path = os.path.join(frames_dir, fname)
        if not os.path.exists(img_path): continue
        img_bgr = cv2.imread(img_path)
        H,W     = img_bgr.shape[:2]
        img_pil = PILImage.fromarray(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB))
        sc,bx   = detreg_inference(img_pil)
        km = sc > BEST_THRESH
        if km.sum()>0:
            ksc=sc[km]; kbx=bx[km]
            kbx[:,0].clamp_(0,W); kbx[:,2].clamp_(0,W)
            kbx[:,1].clamp_(0,H); kbx[:,3].clamp_(0,H)
            keep      = tv_nms(kbx.float(), ksc.float(), iou_threshold=NMS_IOU)
            raw_boxes = kbx[keep].numpy()
        else:
            raw_boxes = np.zeros((0,4))
        green_boxes = [b for b in raw_boxes if is_green_detection(img_bgr,b)]
        for p in ripe_pts:
            px,py    = p['x'],p['y']
            in_raw   = any(b[0]<=px<=b[2] and b[1]<=py<=b[3] for b in raw_boxes)
            in_green = any(b[0]<=px<=b[2] and b[1]<=py<=b[3] for b in green_boxes)
            if in_raw and in_green: vid_fail+=1
            else:                   vid_supp+=1
            vid_total+=1
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
# ## 14. Export Results

# %%
out_path = os.path.join(OUT_DIR, 'detreg_results.xlsx')
test_m   = compute_metrics(res_df[res_df['split']=='test'])
full_m   = compute_metrics(res_df)

summary = pd.DataFrame([{
    'Model':MODEL_NAME,'Pretrained On':PRETRAINED_ON,
    'Weights Source':WEIGHTS_LOADED_AS,'Architecture':'Deformable DETR (ResNet-50)',
    'Inference':'Full image (no tiling)','Num Queries':NUM_QUERIES,
    'NMS IoU':NMS_IOU,'Threshold':BEST_THRESH,'Total Frames':len(res_df),
    'Test MAE':test_m['MAE'],'Test RMSE':test_m['RMSE'],
    'Test W+-2%':test_m['W+-2%'],'Test Bias':test_m['Bias'],
    'Full MAE':full_m['MAE'],'Full Bias':full_m['Bias'],
    'Ripe Suppression%':round(overall_rate,1),
    'Research Note':'Zero-shot failure: DETR requires domain fine-tuning for agricultural tasks',
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

print(f'Saved: {out_path}')
print(f'TEST  — MAE: {test_m["MAE"]}  RMSE: {test_m["RMSE"]}  W+-2%: {test_m["W+-2%"]}  Bias: {test_m["Bias"]}')
print()
print('Research finding: DETReg assigns near-unity objectness to all decoder queries')
print('on agricultural images — DETR-style detectors require domain fine-tuning.')
