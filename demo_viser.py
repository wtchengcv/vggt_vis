# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import time
import threading
import argparse
from typing import List, Optional
import yaml

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2

import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_cmap_func(name: str):
    name = (name or "viridis").lower()
    if name == "viridis":
        cmap = cm.get_cmap('viridis')
    elif name == "turbo":
        cmap = cm.get_cmap('turbo')
    elif name == "jet":
        cmap = cm.get_cmap('jet')
    elif name == "bcy":
        # 蓝→青→黄
        cmap = LinearSegmentedColormap.from_list("bc_y", [(0,0,1),(0,1,1),(1,1,0)], N=256)
    else:
        cmap = cm.get_cmap('viridis')
    def _apply(x01: np.ndarray) -> np.ndarray:
        return (cmap(np.clip(x01, 0, 1))[..., :3] * 255).astype(np.uint8)
    return _apply

def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Here we compute the threshold value based on the current percentage
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server


# Helper functions for sky segmentation


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # Apply sky mask to confidence scores
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf


def _head_reduce_to_bqk(attn: torch.Tensor) -> torch.Tensor:
    """attn: (B,H,Q,K) or (B,Q,K) -> (B,Q,K), float32"""
    if attn.ndim == 4:
        attn = attn.mean(dim=1)
    return attn.float()

def _cm_jet01(x01: np.ndarray) -> np.ndarray:
    return (cm.get_cmap('jet')(np.clip(x01, 0, 1))[..., :3] * 255).astype(np.uint8)

def _cm_blue_cyan_yellow_alt(x01: np.ndarray) -> np.ndarray:
    cmap = cm.get_cmap('turbo')  # 蓝-青-黄-红，但比 jet 更柔和
    return (cmap(np.clip(x01, 0, 1))[..., :3] * 255).astype(np.uint8)

def _cm_blue_cyan_yellow(x01: np.ndarray) -> np.ndarray:
    cmap = cm.get_cmap('viridis')  # 颜色顺序：紫-蓝-青-黄-绿（有点偏绿）
    return (cmap(np.clip(x01, 0, 1))[..., :3] * 255).astype(np.uint8)

def _abs_index_of_special(P: int, f_q: int, tok_type: str, reg_id: int, patch_start_idx: int) -> int:
    """
    返回“全序列绝对索引”的 Query 下标：
      - tok_type='cam'     -> local=0
      - tok_type='reg'     -> local=1+reg_id (reg_id in [0, K-1])
    """
    if tok_type == 'cam':
        local = 0
    elif tok_type == 'reg':
        local = 1 + int(reg_id)
    else:
        raise ValueError("tok_type must be 'cam' or 'reg'")
    return f_q * P + local

# ---------- helpers ----------
def _cm_jet(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (cm.get_cmap('jet')(x01)[..., :3] * 255).astype(np.uint8)  # RGB uint8

def _draw_inset(base_img_bgr: np.ndarray, inset_bgr: np.ndarray, x: int, y: int, alpha: float=1.0, border=True):
    ih, iw = inset_bgr.shape[:2]
    H, W = base_img_bgr.shape[:2]
    x2, y2 = min(x+iw, W), min(y+ih, H)
    roi = base_img_bgr[y:y2, x:x2]
    ins = inset_bgr[:y2-y, :x2-x]
    if alpha < 1.0:
        blended = cv2.addWeighted(roi, 1.0 - alpha, ins, alpha, 0)
    else:
        blended = ins
    base_img_bgr[y:y2, x:x2] = blended
    if border:
        cv2.rectangle(base_img_bgr, (x, y), (x2-1, y2-1), (255,255,255), 1, cv2.LINE_AA)

def _decode_token(q_abs_or_local: int, is_frame: bool, P: int, S: int, K: int, patch_start_idx: int,
                  H_patch: int, W_patch: int):
    """
    返回: (f_q, tok_type, reg_id, (py,px))
    - global 模式: q_abs = [0, S*P)，f_q = q_abs // P
    - frame  模式: q_local = [0, P)，f_q 在可视化时由 batch index 决定（这里先返回 None）
    """
    if is_frame:
        within = q_abs_or_local
        if not (0 <= within < P):
            raise ValueError(f"[decode] frame q_local={within} out of [0,{P})")
        f_q = None  # 在 frame 模式，所属帧 = 当前 batch entry
    else:
        q_abs = q_abs_or_local
        f_q = q_abs // P
        within = q_abs % P
        if not (0 <= f_q < S):
            raise ValueError(f"[decode] global q_abs maps to f_q={f_q} >= S={S}")

    if within == 0:
        return f_q, 'cam', None, None
    elif 1 <= within < patch_start_idx:
        return f_q, 'reg', (within - 1), None
    else:
        lin = within - patch_start_idx
        if not (0 <= lin < H_patch * W_patch):
            raise ValueError(f"[decode] patch lin={lin} out of [0,{H_patch*W_patch})")
        py, px = divmod(lin, W_patch)
        return f_q, 'patch', None, (py, px)

# ---------- main function ----------
def save_attn_heatmaps_from_records(
    images_tensor: torch.Tensor,     # (S,3,H,W) in [0,1]
    attn_records: list,              # List[dict]: {'attn','q_indices','is_frame','P','S','K','patch_start_idx','block_idx'}
    save_dir: str,
    which_frames=None,               # None=全部帧；或 [0,1,2,...]
    prefix: str = 'attn',
    patch_size: int = None,          # 建议传 agg.patch_size；若 None，会用 P 估计网格
    overlay_query: bool = True,      # 在热图上叠加 Query 位置/标签
    draw_query_on_main: bool = True, # 仅当 query 为 patch 且渲染帧==query所属帧时，在主图上画绿框
    draw_inset: bool = False,        # 左上角角标小图
    inset_scale: float = 0.22,
    tok_filter: set = None,           # e.g. {'cam'} / {'reg'} / {'patch'}；None 表示不过滤
    cmap_func=None,
    vmax_percentile=None,
):
    os.makedirs(save_dir, exist_ok=True)
    S_img, _, H_full, W_full = images_tensor.shape
    frames_all = list(range(S_img))
    frames_user = set(frames_all if which_frames is None else [f for f in which_frames if f in frames_all])

    for rec_id, rec in enumerate(attn_records):
        attn = rec['attn']                      # (B, lenQ, N) 或 (B, H, lenQ, N)
        if attn.ndim == 4:                      # 若按头维存，先对 head 做平均
            attn = attn.mean(dim=1)
        attn = attn.cpu()
        Bcur, lenQ, N = attn.shape

        is_frame  = bool(rec['is_frame'])
        P = int(rec['P'])
        S = min(int(rec['S']), S_img)
        K = int(rec['K'])
        patch_start_idx = int(rec['patch_start_idx'])
        blk = int(rec.get('block_idx', -1))
        q_indices = list(rec['q_indices'])

        # 计算 patch 网格
        if patch_size is not None:
            assert H_full % patch_size == 0 and W_full % patch_size == 0, \
                f"({H_full},{W_full}) not divisible by patch_size={patch_size}"
            H_patch = H_full // patch_size
            W_patch = W_full // patch_size
        else:
            num_patches = P - (1 + K)
            # 用图像宽高比估一个 H_patch/W_patch
            ar = W_full / max(1, H_full)
            H_patch = max(1, int(round(np.sqrt(num_patches / max(1e-6, ar)))))
            W_patch = max(1, num_patches // H_patch)
            if H_patch * W_patch != num_patches:
                H_patch, W_patch = num_patches, 1

        # 尺寸 sanity
        if is_frame:
            assert N == P, f"[save] frame-wise: N={N} must equal P={P}"
        else:
            assert N == S * P, f"[save] global: N={N} must equal S*P={S*P}"

        # 遍历每个 query（该 rec 里抓到的所有行）
        for qi_rel, q_sel in enumerate(q_indices):

            # 解码 query 元信息（类型/位置）
            f_q_dec, tok_type, reg_id, patch_rc = _decode_token(
                q_sel, is_frame, P, S, K, patch_start_idx, H_patch, W_patch
            )
            if tok_filter and tok_type not in tok_filter:
                continue

            # 遍历 batch（=S 或 1）
            for b in range(Bcur):
                row = attn[b, qi_rel]  # (N,)

                if is_frame:
                    # 帧内注意力：只能在“自己的帧 b”上画
                    f_entry = b  # 当前 batch entry 对应的帧号
                    if f_entry not in frames_user:
                        continue

                    # 只取本帧的 patch 段
                    vec = row[patch_start_idx:P].numpy()
                    f_render_list = [f_entry]
                    # 在 frame 模式下，“query 所属帧”= 当前 entry
                    f_q = f_entry

                else:
                    # global 注意力：可以选择任何帧 f 来可视化
                    f_render_list = sorted(frames_user)
                    f_q = f_q_dec  # 解码得到的 query 所属帧
                    # 取 f 帧的 patch 段时再切片

                # 逐帧渲染
                for f in f_render_list:
                    if is_frame:
                        grid_vec = vec
                    else:
                        base = f * P
                        grid_vec = row[base + patch_start_idx : base + P].numpy()

                    grid = grid_vec.reshape(H_patch, W_patch)
                    denom = (grid.max() - grid.min()) + 1e-8
                    heat = (grid - grid.min()) / denom
                    heat = cv2.resize(heat, (W_full, H_full), interpolation=cv2.INTER_CUBIC)
                    heat_rgb = _cm_jet(heat)  # RGB

                    # 叠加到第 f 帧
                    img_rgb = (images_tensor[f].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                    out_rgb = cv2.addWeighted(img_rgb, 0.6, heat_rgb, 0.4, 0)
                    out_bgr = out_rgb[:, :, ::-1].copy()

                    # 在主图上标 query 位置（仅 patch 且 f == f_q）
                    if overlay_query and draw_query_on_main and (tok_type == 'patch') and (f == f_q) and (patch_size is not None):
                        py, px = patch_rc
                        x0 = int(px * patch_size); y0 = int(py * patch_size)
                        x1 = int(min((px+1)*patch_size, W_full)) - 1
                        y1 = int(min((py+1)*patch_size, H_full)) - 1
                        cv2.rectangle(out_bgr, (x0,y0), (x1,y1), (0,255,0), 2)
                        cv2.circle(out_bgr, ((x0+x1)//2, (y0+y1)//2), max(2, patch_size//8), (0,255,0), -1)
                        cv2.putText(out_bgr, f"Q patch ({py},{px}) f{f_q}", (x0, max(20,y0-8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

                    # 可选：左上角 inset（默认关）
                    if overlay_query and draw_inset:
                        # 用 query 所属帧的原图做 inset
                        inset_rgb = (images_tensor[f_q].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8) if (not is_frame) else \
                                    (images_tensor[f_q].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                        inset_bgr = inset_rgb[:, :, ::-1].copy()
                        label = {'cam':'CAM', 'reg': f'REG#{reg_id}', 'patch': f'PATCH@{patch_rc}'}[tok_type]
                        cv2.putText(inset_bgr, f"Q:{label} f{f_q}", (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2, cv2.LINE_AA)
                        long_side = max(H_full, W_full)
                        scale = inset_scale * (long_side / max(inset_bgr.shape[0], inset_bgr.shape[1]))
                        inset_res = cv2.resize(inset_bgr, (int(inset_bgr.shape[1]*scale), int(inset_bgr.shape[0]*scale)))
                        _draw_inset(out_bgr, inset_res, x=10, y=10, alpha=1.0, border=True)

                    # 命名：global 用 qabs；frame 用 qlocal
                    tok_name = {'cam':'cam', 'reg': f'reg{reg_id}', 'patch':'patch'}[tok_type]
                    if is_frame:
                        qtag = f"qlocal{q_sel}"
                        fqt  = f"fq{f_q}"
                    else:
                        qtag = f"qabs{q_sel}"
                        fqt  = f"fq{f_q}"

                    # 保留 batch entry id 以便排障（frame 模式下 = f_q = b）
                    fname = os.path.join(
                        save_dir,
                        f"{prefix}_blk{blk}_rec{rec_id}_b{b}_f{f}_{qtag}_{fqt}_{tok_name}.png"
                    )
                    cv2.imwrite(fname, out_bgr)
                    # print("saved:", fname)

def save_frame_to_frame_patchmap_auto(
    rec: dict,
    f_q: int,                 # 源帧：取该帧所有 patch 作为行
    f_t: int,                 # 目标帧：取该帧所有 patch 作为列（frame 模式会强制 == f_q）
    save_dir: str,            # 目录（函数会自动命名文件名）
    save_path: str = None,    # 若提供则用你给的完整路径
    row_norm: bool = True,
    vmax_percentile: float = 99.5,
    draw_ticks: bool = True,
    cmap_func=None
):
    """
    统一接口：自动识别 rec['is_frame']。
    - global 记录：可视化 (f_q 的所有 patch) → (f_t 的所有 patch) 的注意力子矩阵。
      需要本次抓到“全体 Query”（Q_total == S*P）；否则会报错提示把 capture_queries='all'。
    - frame  记录：只可视化 (f_q 的所有 patch) → (同一帧的所有 patch)，即 f_t 必须等于 f_q。
      注意很多实现把每帧拆到 batch 维（B≈S），本函数按此惯例处理。
    """
    # 基本元信息
    is_frame = bool(rec.get('is_frame', False))
    P   = int(rec['P'])
    S   = int(rec['S'])
    Krg = int(rec['K'])
    p0  = int(rec['patch_start_idx'])  # = 1 + Krg
    blk = rec.get('block_idx', -1)

    A = rec['attn']              # (B,H?,Q,K) or (B,Q,K)
    A = _head_reduce_to_bqk(A)   # (B,Q,K)
    if A.numel() == 0:
        raise RuntimeError("[ff-patchmap] rec['attn'] 为空，请检查 hook 是否生效。")

    Bcur, Q_total, K_total = A.shape

    # 路径命名
    prefix = "frame" if is_frame else "global"
    if save_path is None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{prefix}_blk{blk}_fq{f_q}_ft{f_t}_patchmap.png")

    # 取子矩阵（patch×patch）
    if is_frame:
        # 仅帧内注意力：Q=K=P；只能 f_t==f_q
        if f_t != f_q:
            raise AssertionError("[ff-patchmap] frame attention 不支持跨帧，可视化时应令 f_t == f_q。")
        if not (Q_total == P and K_total == P):
            raise AssertionError(f"[ff-patchmap] frame 记录应为 Q=K=P={P}，实际 Q={Q_total},K={K_total}")

        # 行列都是该帧的 patch 段
        r0, r1 = p0, P
        c0, c1 = p0, P
        sub = A[f_q if Bcur >= (f_q+1) else 0, r0:r1, c0:c1].cpu().numpy()  # (Hp*Wp, Hp*Wp)

    else:
        # 全局注意力：期望 Q_total=K_total=S*P（抓全体 Query）
        if not (Q_total == S*P and K_total == S*P):
            raise AssertionError(
                f"[ff-patchmap] global 记录期望 Q=K=S*P={S*P}；实际 Q={Q_total},K={K_total}。"
                "大概率因未抓全体 Query（如 capture_queries='indices' / 'cam+reg'）。"
                "请在这次可视化前用 capture_queries='all' 跑一遍。"
            )
        # 源帧的 patch 行范围，目标帧的 patch 列范围
        r0, r1 = f_q * P + p0, f_q * P + P
        c0, c1 = f_t * P + p0, f_t * P + P
        sub = A[0, r0:r1, c0:c1].cpu().numpy()  # (Hp*Wp, Hp*Wp)

    if sub.size == 0:
        raise AssertionError("[ff-patchmap] 子矩阵为空，请核对 f_q/f_t 与元信息 p0,P,S。")

    # 归一 & 对比分位数拉伸
    if row_norm:
        sub = sub - sub.min(axis=1, keepdims=True)
        denom = sub.max(axis=1, keepdims=True)
        denom[denom < 1e-8] = 1.0
        sub = sub / denom

    # 全局分位数拉伸
    vmax = np.percentile(sub, vmax_percentile)
    sub = np.clip(sub / (vmax + 1e-8), 0.0, 1.0)

    # 伪彩
    img = _cm_blue_cyan_yellow(sub)    # (Qp, Kp, 3) RGB


    # 网格/标题
    if draw_ticks:
        Qp, Kp = img.shape[:2]
        side = int(np.sqrt(Qp) + 0.5)
        step = max(1, side)
        color = (255, 255, 255)
        for t in range(step, Qp, step):
            cv2.line(img, (0, t), (Kp-1, t), color, 1, cv2.LINE_AA)
        for t in range(step, Kp, step):
            cv2.line(img, (t, 0), (t, Qp-1), color, 1, cv2.LINE_AA)

        pad = 30
        canvas = np.ones((img.shape[0]+pad, img.shape[1], 3), dtype=np.uint8) * 20
        canvas[pad:, :, :] = img
        title = f"{prefix}: frame {f_q} patches → frame {f_t} patches (blk={blk})"
        cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
        img = canvas

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img[:, :, ::-1])  # BGR
    print(f"[frame→frame patchmap] saved: {save_path}")

parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images"
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
parser.add_argument("--config", type=str, default="configs/attn_vis.yaml",
                    help="Path to YAML config for attention visualization")

def main():
    """
    Main function for the VGGT demo with viser for 3D visualization.

    This function:
    1. Loads the VGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser

    Command-line arguments:
    --image_folder: Path to folder containing input images
    --use_point_map: Use point map instead of depth-based points
    --background_mode: Run the viser server in background mode
    --port: Port number for the viser server
    --conf_threshold: Initial percentage of low-confidence points to filter out
    --mask_sky: Apply sky segmentation to filter out sky points
    """
    
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    cmap_func = get_cmap_func(cfg["vis"].get("colormap", "viridis"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    # model = VGGT()
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    
    model = VGGT()
    model.load_state_dict(torch.load("/home/wentaocheng/Documents/model_zoo/vggt.pt", map_location="cuda"))

    model.eval()
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = glob.glob(os.path.join(args.image_folder, "*"))
    print(f"Found {len(image_names)} images")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    
    agg = model.aggregator
    agg.capture_attn = True
    agg.attn_records = {'frame': [], 'global': []}

    # capture.which
    which_set = set(cfg["capture"]["which"])
    agg.capture_which = which_set  # 例如 {"global"} 或 {"frame","global"}

    # capture.queries
    agg.capture_queries = cfg["capture"]["queries"]  # "all" | "cam" | "cam+reg" | "indices"
    agg.capture_query_indices = cfg["capture"].get("query_indices", [])

    # capture.block_indices
    bi = cfg["capture"]["block_indices"]
    # 允许 None / 空列表
    agg.capture_block_indices = {
        'global': set(bi.get('global') or []) if 'global' in bi else None,
        'frame':  set(bi.get('frame')  or []) if 'frame'  in bi else None,
    }

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    images_vis = images.squeeze(0).cpu() if images.ndim == 5 else images.cpu()  # (S,3,H,W)
    # 取到 aggregator 和注意力缓存
    frame_recs  = getattr(agg, 'attn_records', {}).get('frame',  [])
    global_recs = getattr(agg, 'attn_records', {}).get('global', [])

    out_dir = cfg["vis"]["out_dir"]
    which_frames_cfg = cfg["vis"]["which_frames"]
    if which_frames_cfg == "all":
        which_frames = list(range(images_vis.shape[0]))
    else:
        which_frames = list(which_frames_cfg)
    
    # 1) 叠图热力图
    if cfg["viz"]["attn_heatmaps"]:
        if global_recs:
            save_attn_heatmaps_from_records(
                images_tensor=images_vis, attn_records=global_recs,
                save_dir=os.path.join(out_dir, "global"),
                which_frames=which_frames,
                prefix="global",
                patch_size=agg.patch_size,
                overlay_query=cfg["vis"]["overlay_query"],
                draw_query_on_main=cfg["vis"]["draw_query_on_main"],
                draw_inset=cfg["vis"]["draw_inset"],
                tok_filter=None,                # 也可做成配置
                # 新增参数：传 cmap
                # 你需要在函数签名里加 cmap_func，默认 viridis
                cmap_func=cmap_func,
                vmax_percentile=cfg["vis"]["vmax_percentile"],
            )
        if frame_recs:
            save_attn_heatmaps_from_records(
                images_tensor=images_vis, attn_records=frame_recs,
                save_dir=os.path.join(out_dir, "frame"),
                which_frames=which_frames,
                prefix="frame",
                patch_size=agg.patch_size,
                overlay_query=cfg["vis"]["overlay_query"],
                draw_query_on_main=cfg["vis"]["draw_query_on_main"],
                draw_inset=cfg["vis"]["draw_inset"],
                cmap_func=cmap_func,
                vmax_percentile=cfg["vis"]["vmax_percentile"],
            )

    # 2) patchmap（帧↔帧子矩阵）
    if cfg["viz"]["patchmap"] and global_recs + frame_recs:
        pairs = cfg["patchmap"].get("pairs", [])
        for pair in pairs:
            prefix = pair["prefix"]      # "global" or "frame"
            blk    = pair["blk"]
            fq     = pair["fq"]
            ft     = pair["ft"]

            src = (global_recs if prefix == "global" else frame_recs)
            cand = [r for r in src if r.get('block_idx', -1) == blk]
            if not cand:
                print(f"[patchmap] no record for {prefix} blk={blk}")
                continue

            save_frame_to_frame_patchmap_auto(
                rec=cand[0],
                f_q=fq, f_t=ft,
                save_dir=os.path.join(out_dir, f"ff_{prefix}"),
                row_norm=cfg["patchmap"]["rows_norm"],
                vmax_percentile=cfg["vis"]["vmax_percentile"],
                draw_ticks=cfg["patchmap"]["draw_ticks"],
                # 同样给它也加 cmap_func 参数
                cmap_func=cmap_func,
            )




    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    print("Starting viser visualization...")

    viser_server = viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )
    print("Visualization complete")


if __name__ == "__main__":
    main()
