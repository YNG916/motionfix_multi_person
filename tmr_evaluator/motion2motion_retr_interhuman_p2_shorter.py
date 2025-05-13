#!/usr/bin/env python3
# tmr_evaluator/motion2motion_retr_interhuman.py

import os, glob
from pathlib import Path
import numpy as np
import torch
import pandas as pd

# 一键加载预训练模型
from src.tmr.load_model import load_model_from_cfg
from tmr_evaluator.motion2motion_retr import read_config

# —— 新增：MotionFix 原生的 135 维预处理依赖 ——
from src.data.features import _get_body_transl_delta_pelv_infer
from src.tmr.data.motionfix_loader import Normalizer

# ——— 1. 构造 135 维 normalizer（路径指向 eval-deps 下的 stats） ———
stats_dir = Path.cwd() / "eval-deps" / "stats" / "humanml3d" / "amass_feats"
normalizer = Normalizer(stats_dir)

# ——— 2. 定义 preprocess_motion_135 ———
def preprocess_motion_135(raw_np: np.ndarray) -> torch.Tensor:
    """
    raw_np: (T,492)
    返回 Tensor (T,135)，已做骨盆增量和平移+6D旋转归一化
    """
    raw = torch.from_numpy(raw_np).float()         # (T,492)
    trans         = raw[:, :3]                     # (T,3)
    global_orient = raw[:, 3:9]                    # (T,6)
    body_pose_6d  = raw[:, 9:9+21*6]               # (T,126)

    # 1) 骨盆平移增量
    trans_delta = _get_body_transl_delta_pelv_infer(global_orient, trans)  # (T,3)
    # 2) 拼接 → (T,135)
    feat = torch.cat([trans_delta, body_pose_6d, global_orient], dim=-1)
    # 3) 同 MotionFix 原生 normalizer
    return normalizer(feat)  # (T,135)

# ——— 3. 构造长度掩码，与 TEMOS 一致 ———
def length_to_mask(lengths, device):
    lengths = torch.tensor(lengths, device=device)
    max_len = int(lengths.max().item())
    mask = torch.arange(max_len, device=device) \
             .expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask  # (batch, max_len)

def main():
    cwd, run_dir = Path.cwd(), Path.cwd()/"eval-deps"
    cfg = read_config(run_dir, return_json=False)
    model = load_model_from_cfg(cfg, ckpt_name="last", eval_mode=True, device="cuda")
    device = next(model.parameters()).device

    # 扫描所有 person2 下 .npy
    data_dir = r"/cvhci/temp/yyang/data_interhuman/InterHuman-Dataset/Dataset/motions_processed/person2"
    files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    keyids = [Path(f).stem for f in files]

    # 切窗参数
    FPS, W_SEC = 20, 4
    W, S = FPS*W_SEC, FPS*W_SEC//2

    # 切窗+135维预处理
    # windows, meta = [], []
    # for key, path in zip(keyids, files):
    #     raw = np.load(path).astype(np.float32)  # (T,492)
    #     T = raw.shape[0]
    #     for start in range(0, T - W + 1, S):
    #         seg = raw[start:start+W]                    # (W,492)
    #         feat135 = preprocess_motion_135(seg)        # Tensor (W,135)
    #         windows.append(feat135)
    #         meta.append((key, start))
    windows = []
    meta    = []
    for key, path in zip(keyids, files):
        raw = np.load(path).astype(np.float32)  # (T,492)
        T   = raw.shape[0]

        # 如果整条序列长度小于一个窗口，直接当一个窗口处理
        if T < W:
            feat = preprocess_motion_135(raw)   # (T,135)
            windows.append(feat)
            meta.append((key, 0))
        else:
        # 否则按半重叠滑窗切分
            for start in range(0, T - W + 1, S):
                seg  = raw[start : start + W]   # (W,492)
                feat = preprocess_motion_135(seg)  # (W,135)
                windows.append(feat)
                meta.append((key, start))

    # 生成 embedding —— 显式 motion 模式
    model.eval()
    embs = []
    with torch.no_grad():
        for feat in windows:
            L = feat.size(0)
            x = feat.unsqueeze(0).to(device)            # (1, L, 135)
            mask = length_to_mask([L], device)          # (1, L)
            motion_dict = {"x": x, "mask": mask, "length": [L]}
            # ← 指定 modality="motion"
            latent = model.encode(motion_dict, modality="motion", sample_mean=True)  # (1, D)
            embs.append(latent.squeeze(0).cpu().numpy())

    embs = np.vstack(embs)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    # 计算余弦相似度 & 取 top-2
    sim_mat = embs.dot(embs.T)
    neighbors = np.argsort(-sim_mat, axis=1)[:, 1:3]

    # 保存 CSV
    records = []
    for i, ((key, start), nbrs) in enumerate(zip(meta, neighbors)):
        for rank, j in enumerate(nbrs, start=1):
            key2, start2 = meta[j]
            records.append({
                "query_keyid": key,
                "query_start": start,
                "neighbor_rank": rank,
                "neighbor_keyid": key2,
                "neighbor_start": start2,
                "sim_score": float(sim_mat[i, j]),
            })
    df = pd.DataFrame(records)
    df.to_csv(cwd/"person2_top2_neighbors.csv", index=False, encoding="utf-8-sig")
    print("Done, saved person2_top2_neighbors.csv")

if __name__ == "__main__":
    main()

