# src/tmr/data/interhuman_loader.py

import glob, os
import numpy as np
import torch
from torch import Tensor
from typing import List
from typing import Dict

class InterHumanLoader:
    def __init__(self, data_dir: str, keyids: List[str] = None):
        """
        data_dir: 存放所有 Person1 npy 的目录
        keyids: 可选，只加载其中一部分（若为 None，就扫描全目录）
        """
        self.data_dir = data_dir
        # 如果没指定 keyids，就遍历所有 .npy 文件名（不含后缀）
        if keyids is None:
            files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
            keyids = [os.path.splitext(os.path.basename(p))[0] for p in files]
        self.keyids = keyids

    def __len__(self):
        # 返回数据集大小，使 len(dataset) 可用
        return len(self.keyids)

    def load_keyid(self, keyid: str) -> Dict:
        """
        与 compute_sim_matrix 接口保持一致，返回 dict：
        {
            'keyid': keyid,
            'motion_source': Tensor[(T,192)],
            'motion_target': Tensor[(T,192)]
        }
        """
        path = os.path.join(self.data_dir, keyid + ".npy")
        raw = np.load(path).astype(np.float32)    # shape=(T,492)

        # preprocess：只保留22个关节位置 + 21个关节6D旋转 → (T,192)
        pos22 = raw[:, :22*3]
        rot21 = raw[:, 62*3:62*3+21*6]
        feat = np.concatenate([pos22, rot21], axis=1)

        # to torch
        feat_t = torch.from_numpy(feat)
        return {
            'keyid': keyid,
            'motion_source': feat_t,
            'motion_target': feat_t,  # motion→motion，直接复用自身
        }

