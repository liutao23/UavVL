# -*- coding: utf-8 -*-
import threading
import cv2
import rasterio
import time
import torch.nn.functional as F
from tqdm import tqdm
import torch
import pickle
import os
import csv
import math
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from rasterio.transform import xy as pix2map  # pixel -> map coords via affine transform
from ultralytics import YOLO
from course_match_models.FSRA.make_model import two_view_net
from fine_match_models.point_module.lightglue import LightGlue
from fine_match_models.point_module.superpoint import SuperPoint
from fine_match_models.point_module.sift import SIFT
from fine_match_models.point_module.disk import DISK
from fine_match_models.point_module.aliked import ALIKED
from fine_match_models.point_module.dog_hardnet import DoGHardNet
from fine_match_models.point_module.utils import rbd
from fine_match_models.point_module.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from course_match_models.samplegeo.models import TimmModel  # Sample4Geo

# 禁用 PyTorch 的自动梯度计算（仍建议用 inference_mode 包裹关键推理段）
torch.set_grad_enabled(False)


def haversine_distance(gps1, gps2) -> float:
    """Calculate the haversine distance between two GPS coordinates. (km)"""
    lat1 = math.radians(gps1[0])
    lon1 = math.radians(gps1[1])
    lat2 = math.radians(gps2[0])
    lon2 = math.radians(gps2[1])
    dlong = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlong / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return c * 6371.0


def tensor_to_numpy(x):
    """确保 tensor 在 CPU 上并转换为 NumPy 数组"""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    raise TypeError(f"Unsupported type {type(x)} for conversion to numpy.")


def to_1d_feature(x):
    """把任意形状的特征(可能是 (1,D)/(D,1)/(1,1,D)/...) 压成 (D,)"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x)
    x = np.squeeze(x)
    return x.reshape(-1).astype(np.float32)


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))



# ======================= 配置区 =======================
retrival_methd = 'Sample4geo'         # [FSRA, Sample4geo]
point_method = 'SuperPoint'           # [SuperPoint, SIFT, DISK, ALIKED, DoGHardNet, LoFTR_fp16, LoFTR_mp]
TOPK = 5                              # 粗检索 TopK
RANSAC_THRESH = 5.0                   # cv2.findHomography RANSAC reproj threshold
MIN_MATCHES_FOR_H = 4                 # 计算单应至少需要的匹配数（不再用它过滤输出，只决定能否估 H）

# OK 质量门槛
MIN_INLIERS_OK = 5
MIN_INLIER_RATIO_OK = 0.20            # inliers / matches
REQUIRE_PROJ_IN_BOUNDS = True         # 投影点必须在 ref 图像范围内（强烈建议 True）

# ========== [NEW] 细匹配参考图增强：旋转/尺度鲁棒性 ==========
REF_ROT_ANGLES = [0, 90, 180, 270]    # 逆时针旋转角度
REF_SCALES = [1.0, 0.75, 1.25]        # 尺度因子（建议先只用 [1.0] 验证，再逐步加入）
AUG_EARLY_STOP_INLIERS = 80           # 达到该内点数就提前停止该 ref 的增强尝试
AUG_EARLY_STOP_RATIO = 0.45           # 或内点率达到该值提前停止
# =====================================================

# scene_num = '01'
max_frames = 100

# drone_video_path = f'/media/liutao/B/论文投稿/两阶段视觉定位/flight/sat_paired/{scene_num}/drone_video.mp4'
# Reference_img_path = f'/media/liutao/B/论文投稿/两阶段视觉定位/flight/sat_paired/{scene_num}/satellite_tif_flat'
drone_video_path = '/home/liutao/video/demo/Satellite/UavVL/demo_data/01/drone_video.mp4'
Reference_img_path = '/home/liutao/video/demo/Satellite/UavVL/demo_data/01/satellite_tif_flat'
object_model_path = 'weights/building_car.pt'
# ======================================================

SHOW_MAX_W = 1280   # 显示窗口最大宽度
SHOW_MAX_H = 720    # 显示窗口最大高度

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("----------开始载入模型权重----------")
t_start = time.time()

# -------- 粗匹配模型 --------
if retrival_methd == 'FSRA':
    course_model_path = 'weights/net_119.pth'
    vit_pretrain_path = 'weights/vit_small_p16_224-15ec54c9.pth'
    # Reference_img_feature_path = f'/media/liutao/B/论文投稿/两阶段视觉定位/flight/sat_paired/{scene_num}/FSRA/Reference_img_feature.pkl'
    Reference_img_feature_path = f'/home/liutao/video/demo/Satellite/UavVL/demo_data/01/FSRA/Reference_img_feature.pkl'

    coarse_model = two_view_net(class_num=6400, block=1, pretrain_path=vit_pretrain_path)
    coarse_model.load_state_dict(torch.load(course_model_path, map_location='cpu'), strict=False)
    coarse_model = coarse_model.eval().to(device)

    Transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

elif retrival_methd == 'Sample4geo':
    course_model_path = '/home/liutao/video/demo/Satellite/Visual_Localization/weights/weights_end.pth'
    # Reference_img_feature_path = f'/media/liutao/B/论文投稿/两阶段视觉定位/flight/sat_paired/{scene_num}/Sample4geo/Reference_img_feature.pkl'
    Reference_img_feature_path = f'/home/liutao/video/demo/Satellite/UavVL/demo_data/01/Sample4geo/Reference_img_feature.pkl'

    coarse_model = TimmModel()
    coarse_model.load_state_dict(torch.load(course_model_path, map_location='cpu'), strict=False)
    coarse_model = coarse_model.to(device).eval()

    Transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
else:
    raise ValueError("未指定 coarse retrival_methd")

# -------- 细匹配模型 --------
if point_method == 'SuperPoint':
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)

elif point_method == 'DISK':
    extractor = DISK(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='disk').eval().to(device)

elif point_method == 'ALIKED':
    extractor = ALIKED(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='aliked').eval().to(device)

elif point_method == 'SIFT':
    extractor = SIFT(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='sift').eval().to(device)

elif point_method == 'DoGHardNet':
    extractor = DoGHardNet(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='doghardnet').eval().to(device)

elif 'LoFTR' in point_method:
    model_type = 'full'
    if model_type == 'full':
        _default_cfg = deepcopy(full_default_cfg)
    else:
        _default_cfg = deepcopy(opt_default_cfg)

    if 'mp' in point_method:
        _default_cfg['mp'] = True
    elif 'fp16' in point_method:
        _default_cfg['half'] = True

    matcher = LoFTR(config=_default_cfg)
    ckpt = torch.load(r"fine_match_models/point_module/weight/eloftr_outdoor.ckpt", map_location='cpu')
    matcher.load_state_dict(ckpt['state_dict'])
    matcher = reparameter(matcher)
    if 'fp16' in point_method:
        matcher = matcher.half()
    matcher = matcher.eval().to(device)
else:
    raise ValueError("未指定细匹配模型的方法，请指定 point_method")

# -------- 目标检测模型 --------
model_YOLO = YOLO(object_model_path).eval().to(device)

print("----------载入模型权重结束----------\n"
      f"加载模型耗时 {time.time()-t_start:.3f} s")

# 细匹配图像预处理
Transform_ = transforms.ToTensor()

# 视频输入
cap = cv2.VideoCapture(drone_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
print(f"Total frames: {max_frames}  Frame width: {frame_width}  Frame height: {frame_height}")


# output_dir = f'/media/liutao/B/论文投稿/两阶段视觉定位/flight/sat_paired/{scene_num}/Sample4geo'
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, 'localization_results.csv')
out = cv2.VideoWriter(os.path.join(output_dir, 'drone_localization_visual.mp4'),
                      fourcc, 30.0, (frame_width, frame_height))

print("----------开始执行无人机视频帧定位----------")
start_time = time.time()

# 参考特征
if not os.path.exists(Reference_img_feature_path):
    raise FileNotFoundError(f"未找到特征文件: {Reference_img_feature_path}，请预先提取保存")

with open(Reference_img_feature_path, 'rb') as f:
    Reference_features = pickle.load(f)

# ======================================================================
# ref_mat/ref_files 矩阵化，并归一化一次（用于 GEMM TopK）
# ======================================================================
ref_files = list(Reference_features.keys())  # stable order
ref_vecs = [to_1d_feature(Reference_features[f]) for f in ref_files]
ref_mat = np.stack(ref_vecs, axis=0)  # (N, D)
ref_mat = torch.from_numpy(ref_mat).to(device=device, dtype=torch.float32)
ref_mat = F.normalize(ref_mat, dim=1)  # (N, D)
# ======================================================================

# ======================================================================
# 缓存 GeoTIFF 的 transform/crs/width/height，并用 rasterio.transform 做像素->经纬度
# ======================================================================
try:
    from pyproj import Transformer
except ImportError:
    Transformer = None

geo_cache = {}  # key: ref_path -> dict(crs, transform, width, height)
_transformer_cache = {}  # key: str(crs) -> Transformer


def get_ref_geo(ref_path: str):
    if ref_path in geo_cache:
        return geo_cache[ref_path]
    with rasterio.open(ref_path) as src:
        info = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }
    geo_cache[ref_path] = info
    return info


def pixel_to_wgs84(ref_path: str, x: float, y: float):
    """
    将 ref 图像像素坐标 (x,y) -> WGS84(lat, lon)
    x: col, y: row
    """
    info = get_ref_geo(ref_path)
    crs = info["crs"]
    transform_aff = info["transform"]

    # rasterio 的 row/col 对应 y/x
    map_x, map_y = pix2map(transform_aff, rows=float(y), cols=float(x), offset="center")

    # EPSG:4326 直接返回
    if crs is not None and crs.to_epsg() == 4326:
        lon, lat = map_x, map_y
        return float(lat), float(lon)

    # 其它 CRS 需要投影到 EPSG:4326
    if Transformer is None:
        raise RuntimeError("pyproj 未安装，无法进行 CRS->EPSG:4326 转换。请安装：pip install pyproj")

    key = str(crs)
    if key not in _transformer_cache:
        _transformer_cache[key] = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    lon, lat = _transformer_cache[key].transform(map_x, map_y)
    return float(lat), float(lon)


def project_drone_center_to_ref(H: np.ndarray, drone_w: int, drone_h: int):
    """
    用 H: drone->ref，把 drone 图像中心点投影到 ref 图像像素坐标
    返回 (cx, cy) 或 None
    """
    if H is None:
        return None
    dc = np.array([drone_w / 2.0, drone_h / 2.0, 1.0], dtype=np.float64)
    rp = H @ dc
    if abs(float(rp[2])) < 1e-9:
        return None
    cx = float(rp[0] / rp[2])
    cy = float(rp[1] / rp[2])
    return cx, cy
# ======================================================================


# ======================= [NEW] ref 旋转/缩放增强工具 =======================
def _apply_ref_aug(ref_pil: Image.Image, angle_ccw: int, scale: float):
    """
    对参考图做 旋转(逆时针) + 缩放，返回增强后的 ref_pil_aug 以及 meta
    meta 用于把增强图坐标系下的点逆变换回原始参考图坐标系
    """
    assert angle_ccw in (0, 90, 180, 270)
    assert scale > 0

    w0, h0 = ref_pil.size

    # 先旋转（逆时针）
    if angle_ccw == 0:
        ref_rot = ref_pil
        wr, hr = w0, h0
    elif angle_ccw == 90:
        ref_rot = ref_pil.transpose(Image.Transpose.ROTATE_90)  # CCW 90
        wr, hr = h0, w0
    elif angle_ccw == 180:
        ref_rot = ref_pil.transpose(Image.Transpose.ROTATE_180)
        wr, hr = w0, h0
    else:  # 270
        ref_rot = ref_pil.transpose(Image.Transpose.ROTATE_270)  # CCW 270
        wr, hr = h0, w0

    # 再缩放
    if abs(scale - 1.0) < 1e-9:
        ref_aug = ref_rot
        ws, hs = wr, hr
    else:
        ws = max(2, int(round(wr * scale)))
        hs = max(2, int(round(hr * scale)))
        ref_aug = ref_rot.resize((ws, hs), resample=Image.BILINEAR)

    meta = {
        "w0": float(w0), "h0": float(h0),     # 原图尺寸
        "wr": float(wr), "hr": float(hr),     # 旋转后尺寸（未缩放）
        "ws": float(ws), "hs": float(hs),     # 增强图最终尺寸（含缩放）
        "angle": int(angle_ccw),
        "scale": float(scale),
    }
    return ref_aug, meta


def _inv_map_points_to_original_ref(pts_xy: np.ndarray, meta: dict) -> np.ndarray:
    """
    将增强参考图坐标系下的点 (x,y) 逆变换回 原始参考图坐标系 (x0,y0)
    pts_xy: (N,2) float
    """
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
    angle = meta["angle"]
    scale = meta["scale"]
    w0, h0 = meta["w0"], meta["h0"]

    # 1) 先逆缩放：增强图 -> 旋转图坐标
    if abs(scale - 1.0) > 1e-12:
        pts = pts / float(scale)

    x = pts[:, 0].copy()
    y = pts[:, 1].copy()

    # 2) 再逆旋转：旋转图 -> 原图
    # 增强定义：原图逆时针旋转 angle 得到旋转图
    if angle == 0:
        x0, y0 = x, y
    elif angle == 90:
        # forward: x' = y0, y' = (w0-1-x0)
        # inverse:
        x0 = (w0 - 1.0) - y
        y0 = x
    elif angle == 180:
        x0 = (w0 - 1.0) - x
        y0 = (h0 - 1.0) - y
    else:  # 270
        # forward: x' = (h0-1-y0), y' = x0
        # inverse:
        x0 = y
        y0 = (h0 - 1.0) - x

    out = np.stack([x0, y0], axis=1).astype(np.float32)
    return out
# ======================================================================


def fine_match_and_homography(drone_pil: Image.Image, ref_pil: Image.Image):
    """
    对给定 drone/ref 图做细匹配 + 参考图旋转/尺度增强 + RANSAC Homography
    返回: best_H, best_inliers, best_matches, aux(dict)
      - best_H 是 drone(原始尺寸) -> ref(原始尺寸) 的单应
      - aux 包含使用的 angle/scale
    """
    # 确保 RGB
    if ref_pil.mode == "RGBA":
        ref_pil = ref_pil.convert("RGB")
    if drone_pil.mode == "RGBA":
        drone_pil = drone_pil.convert("RGB")

    best_H = None
    best_inliers = 0
    best_matches = 0
    best_ratio = 0.0
    best_aux = {"angle": 0, "scale": 1.0}

    for ang in REF_ROT_ANGLES:
        for sc in REF_SCALES:
            ref_aug, meta = _apply_ref_aug(ref_pil, ang, sc)

            # ============================
            # 1) 提取匹配点
            # ============================
            if 'LoFTR' in point_method:
                # LoFTR：映射 mkpts 回原始 drone/ref_aug 尺寸，保证 H 在原始尺度上估计
                d0 = drone_pil.convert('L')
                r0 = ref_aug.convert('L')

                w_d, h_d = d0.size
                w_r, h_r = r0.size

                img0_raw_w = int(0.5 * w_d) // 32 * 32
                img0_raw_h = int(0.5 * h_d) // 32 * 32
                img1_raw_w = int(0.5 * w_r) // 32 * 32
                img1_raw_h = int(0.5 * h_r) // 32 * 32

                if img0_raw_w < 32 or img0_raw_h < 32 or img1_raw_w < 32 or img1_raw_h < 32:
                    continue

                img0_raw = d0.resize((img0_raw_w, img0_raw_h))
                img1_raw = r0.resize((img1_raw_w, img1_raw_h))

                img0_np = np.array(img0_raw, dtype=np.float32) / 255.0
                img1_np = np.array(img1_raw, dtype=np.float32) / 255.0
                img0_tensor = torch.from_numpy(img0_np).unsqueeze(0).unsqueeze(0).to(device)
                img1_tensor = torch.from_numpy(img1_np).unsqueeze(0).unsqueeze(0).to(device)

                if 'fp16' in point_method:
                    img0_tensor = img0_tensor.half()
                    img1_tensor = img1_tensor.half()

                batch = {'image0': img0_tensor, 'image1': img1_tensor}

                with torch.inference_mode():
                    if 'mp' in point_method:
                        with torch.autocast(enabled=True, device_type='cuda'):
                            matcher(batch)
                    else:
                        matcher(batch)

                mk0 = batch.get('mkpts0_f', None)
                mk1 = batch.get('mkpts1_f', None)
                if mk0 is None or mk1 is None:
                    continue

                drone_points = mk0.detach().cpu().numpy()
                ref_points_aug = mk1.detach().cpu().numpy()

                mconf = batch.get('mconf', None)
                if mconf is not None:
                    mconf = mconf.detach().cpu().numpy()
                    top_k = 512
                    if len(mconf) > top_k:
                        idx = np.argsort(mconf)[::-1][:top_k]
                        drone_points = drone_points[idx]
                        ref_points_aug = ref_points_aug[idx]

                # 将点从 LoFTR 输入尺寸映射回原始 drone/ref_aug 尺寸
                sx_d = float(w_d) / float(img0_raw_w)
                sy_d = float(h_d) / float(img0_raw_h)
                sx_r = float(w_r) / float(img1_raw_w)
                sy_r = float(h_r) / float(img1_raw_h)

                drone_points[:, 0] *= sx_d
                drone_points[:, 1] *= sy_d
                ref_points_aug[:, 0] *= sx_r
                ref_points_aug[:, 1] *= sy_r

                drone_np = np.asarray(drone_points, dtype=np.float64).reshape(-1, 2)
                ref_aug_np = np.asarray(ref_points_aug, dtype=np.float64).reshape(-1, 2)

            else:
                drone_t = Transform_(drone_pil).to(device)
                ref_t = Transform_(ref_aug).to(device)

                with torch.inference_mode():
                    drone_feats = extractor.extract(drone_t)
                    ref_feats = extractor.extract(ref_t)
                    matches01 = matcher({'image0': drone_feats, 'image1': ref_feats})

                drone_feat, ref_feat, matches01 = [rbd(x) for x in [drone_feats, ref_feats, matches01]]
                matches = matches01.get('matches', None)
                if matches is None or len(matches) == 0:
                    continue

                drone_points = drone_feat['keypoints'][matches[..., 0]]
                ref_points_aug = ref_feat['keypoints'][matches[..., 1]]

                drone_np = np.asarray(tensor_to_numpy(drone_points), dtype=np.float64).reshape(-1, 2)
                ref_aug_np = np.asarray(tensor_to_numpy(ref_points_aug), dtype=np.float64).reshape(-1, 2)

            matches_count = int(min(len(drone_np), len(ref_aug_np)))
            if matches_count < MIN_MATCHES_FOR_H:
                continue

            # ============================
            # 2) ref_aug 点逆变换回原始 ref 坐标系
            # ============================
            ref_np = _inv_map_points_to_original_ref(ref_aug_np, meta).astype(np.float64)

            # ============================
            # 3) RANSAC Homography（drone原始 -> ref原始）
            # ============================
            H, mask = cv2.findHomography(drone_np, ref_np, cv2.RANSAC, RANSAC_THRESH)
            if H is None or mask is None:
                continue

            mask = mask.reshape(-1).astype(bool)
            inliers_count = int(mask.sum())
            inlier_ratio = float(inliers_count) / float(matches_count) if matches_count > 0 else 0.0

            # 选最优：inliers 最大；平局看 ratio；再平局看 matches
            better = False
            if inliers_count > best_inliers:
                better = True
            elif inliers_count == best_inliers and inlier_ratio > best_ratio:
                better = True
            elif inliers_count == best_inliers and abs(inlier_ratio - best_ratio) < 1e-12 and matches_count > best_matches:
                better = True

            if better:
                best_H = H
                best_inliers = inliers_count
                best_matches = matches_count
                best_ratio = inlier_ratio
                best_aux = {"angle": int(ang), "scale": float(sc)}

            # 提前停止：该 ref 在某个增强上已经非常好
            if (inliers_count >= AUG_EARLY_STOP_INLIERS) or (inlier_ratio >= AUG_EARLY_STOP_RATIO):
                return best_H, best_inliers, best_matches, best_aux

    return best_H, best_inliers, best_matches, best_aux


# CSV 表头：若不存在则写
write_header = not os.path.exists(csv_path)
if write_header:
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Frame', 'Status', 'BestRef', 'Inliers', 'Matches', 'InlierRatio', 'ProjCx', 'ProjCy', 'Latitude', 'Longitude'
        ])

# 主循环（失败也写 CSV/视频）
for frame_idx in range(1, max_frames + 1):
    ret, frame = cap.read()
    if not ret:
        print(f"[Frame {frame_idx:05d}] 视频读取失败，提前结束。")
        break

    # YOLO 检测
    results = model_YOLO.predict(source=frame, verbose=False)

    # 粗检索：TopK（矩阵化 GEMM）
    drone_img_pil = Image.fromarray(frame)
    drone_tensor = Transform(drone_img_pil).unsqueeze(0).to(device)

    with torch.inference_mode():
        drone_feat, _ = coarse_model(drone_tensor, None)
        drone_feat = drone_feat.reshape(drone_feat.shape[0], -1)  # -> (1, D)

        drone_vec = F.normalize(drone_feat, dim=1)  # (1, D)
        sims = (drone_vec @ ref_mat.T).squeeze(0)  # (N,)

        topk_vals, topk_idx = torch.topk(sims, k=TOPK, largest=True)
        top_matches = [(ref_files[i], float(topk_vals[j]))
                       for j, i in enumerate(topk_idx.tolist())]

    # 对 TopK 候选逐个细匹配，选 inliers 最大者（平局看相似度）
    best = {
        "ref_file": None,
        "ref_path": None,
        "sim": -1e9,
        "H": None,
        "inliers": -1,
        "matches": 0,
        "inlier_ratio": 0.0,
        "proj_center": None,   # (cx, cy) in ref image
        "aug_angle": 0,
        "aug_scale": 1.0,
    }

    for ref_file, sim in top_matches:
        ref_path = os.path.join(Reference_img_path, ref_file)

        # 读取参考图（PIL）供细匹配
        try:
            ref_pil = Image.open(ref_path)
            if ref_pil.mode == "RGBA":
                ref_pil = ref_pil.convert("RGB")
        except Exception:
            continue

        H, inliers, matches_cnt, aux = fine_match_and_homography(drone_img_pil, ref_pil)
        ratio = float(inliers) / float(matches_cnt) if matches_cnt > 0 else 0.0

        # 选最优：inliers 最大；inliers 相同则 sim 更大
        better = False
        if inliers > best["inliers"]:
            better = True
        elif inliers == best["inliers"] and sim > best["sim"]:
            better = True

        if better:
            best.update({
                "ref_file": ref_file,
                "ref_path": ref_path,
                "sim": sim,
                "H": H,
                "inliers": inliers,
                "matches": matches_cnt,
                "inlier_ratio": ratio,
                "aug_angle": int(aux.get("angle", 0)) if aux is not None else 0,
                "aug_scale": float(aux.get("scale", 1.0)) if aux is not None else 1.0,
            })

    # 输出经纬度：像素->经纬度 + drone center 投影 + OK 质量门槛
    status = "FAIL"
    latitude, longitude = None, None

    if best["ref_path"] is not None and best["H"] is not None:
        if best["inliers"] >= MIN_INLIERS_OK:
            if best["matches"] > 0 and (best["inliers"] / best["matches"]) >= MIN_INLIER_RATIO_OK:
                proj = project_drone_center_to_ref(best["H"], frame_width, frame_height)
                if proj is not None:
                    cx, cy = proj
                    info = get_ref_geo(best["ref_path"])
                    ref_w = info["width"]
                    ref_h = info["height"]
                    in_bounds = (0.0 <= cx < ref_w) and (0.0 <= cy < ref_h)

                    if (not REQUIRE_PROJ_IN_BOUNDS) or in_bounds:
                        latitude, longitude = pixel_to_wgs84(best["ref_path"], cx, cy)
                        best["proj_center"] = (cx, cy)
                        status = "OK"

    # 画框 + 写字
    if status == "OK":
        cv2.putText(frame, f"[{status}] Lat:{latitude:.6f} Lon:{longitude:.6f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"[{status}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    info2 = f"TopK={TOPK} Inliers={max(best['inliers'],0)} Matches={best['matches']} Ratio={best['inlier_ratio']:.2f}"
    cv2.putText(frame, info2, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if best["ref_file"] is not None:
        cv2.putText(frame, f"Ref:{best['ref_file']}  aug=({best['aug_angle']}deg, x{best['aug_scale']:.2f})",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    # 画检测框
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model_YOLO.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 写视频（无论成功失败都写）
    out.write(frame)

    # 可视化（可选）
    h, w = frame.shape[:2]
    scale = min(SHOW_MAX_W / w, SHOW_MAX_H / h, 1.0)  # 只缩小，不放大
    show_frame = frame if scale >= 1.0 else cv2.resize(frame, (int(w * scale), int(h * scale)),
                                                       interpolation=cv2.INTER_AREA)
    cv2.imshow("YOLOv8 Detection", show_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # 打印
    if status == "OK":
        cx, cy = best["proj_center"] if best["proj_center"] is not None else (None, None)
        print(f"[Frame {frame_idx:05d}] status={status:4s} "
              f"lat={latitude:.6f} lon={longitude:.6f} "
              f"ref={best['ref_file']} sim={best['sim']:.4f} "
              f"matches={best['matches']} inliers={best['inliers']} ratio={best['inlier_ratio']:.3f} "
              f"proj=({cx:.1f},{cy:.1f}) "
              f"aug=({best['aug_angle']}deg,x{best['aug_scale']:.2f})")
    else:
        print(f"[Frame {frame_idx:05d}] status={status:4s} "
              f"lat=None lon=None "
              f"ref={best['ref_file']} sim={best['sim']:.4f} "
              f"matches={best['matches']} inliers={max(best['inliers'],0)} ratio={best['inlier_ratio']:.3f} "
              f"aug=({best['aug_angle']}deg,x{best['aug_scale']:.2f})")

    # 追加写 CSV（无论成功失败都写）
    proj_cx, proj_cy = "", ""
    if best.get("proj_center") is not None:
        proj_cx, proj_cy = best["proj_center"]

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            frame_idx,
            status,
            best["ref_file"] if best["ref_file"] is not None else "",
            max(best["inliers"], 0),
            best["matches"],
            float(best["inlier_ratio"]),
            proj_cx,
            proj_cy,
            latitude if latitude is not None else "",
            longitude if longitude is not None else "",
        ])

# 平均耗时
total_processed = frame_idx if 'frame_idx' in locals() else max_frames
elapsed = time.time() - start_time
print(f'平均定位一帧耗时约为：{elapsed / max(1, total_processed):.4f} 秒')

cap.release()
out.release()
cv2.destroyAllWindows()
