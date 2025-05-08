#!/usr/bin/env python3
import os
import sys
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio

# —— 加入 MotionFix 源码到模块搜索路径 —— 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from utils.smpl_body_utils import get_smpl_skeleton

# 使用 SMPLH
from smplx import SMPLH


def render_motion_cpu(path_to_npy: str,
                      model_folder: str,
                      out_dir: str = 'render_output',
                      gif_name: str = 'motion.gif',
                      mp4_name: str = None,
                      fps: int = 25):
    """
    1) Load .npy sample (must contain dict['pose'], shape (T,3+3J))
    2) Auto-detect SMPLH_NEUTRAL.npz under model_folder or model_folder/smplh
    3) Split pose → transl + global_orient + body_pose, zero hand poses
    4) SMPLH forward kinematics → joints (T, J, 3)
    5) CPU-only render each frame to frames/
    6) Synthesize GIF/MP4 under out_dir
    """
    # 1) prepare output directories
    os.makedirs(out_dir, exist_ok=True)
    frame_dir = os.path.join(out_dir, 'frames')
    os.makedirs(frame_dir, exist_ok=True)

    # 2) load sample
    data = np.load(path_to_npy, allow_pickle=True).item()
    if 'pose' not in data:
        raise KeyError(f"'pose' not found in {path_to_npy}")
    pose = data['pose']                # (T, 3+3J)
    T, D = pose.shape
    J = (D - 3) // 3                   # number of SMPL body joints

    # split translation and all-joint axis-angle
    transl = torch.from_numpy(pose[:, :3]).float()     # (T,3)
    rots   = torch.from_numpy(pose[:, 3:]).float()     # (T,3J)

    # 3) split into SMPLH inputs
    global_orient = rots[:, :3]                       # (T,3)
    # flatten body_pose to (T, (J-1)*3)
    body_pose     = rots[:, 3:]                       # (T, (J-1)*3)
    # zero hand poses, flatten to (T, 15*3)
    left_hand     = torch.zeros((T, 15*3), dtype=torch.float32)
    right_hand    = torch.zeros((T, 15*3), dtype=torch.float32)
    # average shape
    betas         = torch.zeros((T, 10), dtype=torch.float32)

    # 4) locate SMPLH model path
    fn          = 'SMPLH_NEUTRAL.npz'
    cand1       = os.path.join(model_folder, fn)
    cand2       = os.path.join(model_folder, 'smplh', fn)
    if   os.path.exists(cand1):
        smplh_dir = model_folder
    elif os.path.exists(cand2):
        smplh_dir = os.path.join(model_folder, 'smplh')
    else:
        raise FileNotFoundError(
            f"Cannot find {fn} under {model_folder} or {model_folder}/smplh"
        )

    # 5) SMPLH forward kinematics
    smplh = SMPLH(
        model_path=smplh_dir,
        gender='neutral',
        ext='npz'
    )
    out   = smplh(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
        left_hand_pose=left_hand,
        right_hand_pose=right_hand,
        transl=transl
    )
    joints = out.joints.detach().cpu().numpy()         # (T, J, 3)

    # 6) get skeleton edges
    edges = get_smpl_skeleton()                        # (n_edges,2)

    # 7) render each frame
    frame_paths = []
    for t in range(T):
        pts = joints[t]                                # (J,3)
        fig = plt.figure(figsize=(4,4), dpi=100)
        ax  = fig.add_subplot(111, projection='3d')

        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=10)
        for i, j in edges:
            ax.plot(
                [pts[i,0], pts[j,0]],
                [pts[i,1], pts[j,1]],
                [pts[i,2], pts[j,2]],
                lw=2
            )

        ax.view_init(elev=15, azim=45)
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        fp = os.path.join(frame_dir, f'frame_{t:03d}.png')
        plt.savefig(fp, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        frame_paths.append(fp)

    # 8) synthesize GIF
    gif_path = os.path.join(out_dir, gif_name)
    with imageio.get_writer(gif_path, mode='I', fps=fps) as writer:
        for fp in frame_paths:
            writer.append_data(imageio.imread(fp))
    print(f"✅ GIF saved to {gif_path}")

    # 9) optional MP4
    if mp4_name:
        mp4_path = os.path.join(out_dir, mp4_name)
        with imageio.get_writer(mp4_path, mode='I', fps=fps, codec='libx264') as writer:
            for fp in frame_paths:
                writer.append_data(imageio.imread(fp))
        print(f"✅ MP4 saved to {mp4_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pure-CPU render of MotionFix SMPLH sample → GIF/MP4'
    )
    parser.add_argument('--path',         required=True,
                        help='.npy sample path (must contain dict["pose"])')
    parser.add_argument('--model_folder', required=True,
                        help='Folder containing SMPLH_NEUTRAL.npz (or its parent)')
    parser.add_argument('--out_dir',      default='render_output',
                        help='Output dir for frames/, GIF, MP4')
    parser.add_argument('--gif',          dest='gif_name', default='motion.gif',
                        help='GIF filename')
    parser.add_argument('--mp4',          dest='mp4_name', default=None,
                        help='Optional MP4 filename')
    parser.add_argument('--fps',          type=int, default=25,
                        help='Frames per second')
    args = parser.parse_args()

    render_motion_cpu(
        path_to_npy   = args.path,
        model_folder  = args.model_folder,
        out_dir       = args.out_dir,
        gif_name      = args.gif_name,
        mp4_name      = args.mp4_name,
        fps           = args.fps
    )
