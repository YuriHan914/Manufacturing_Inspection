"""
2D-3D multi-view fusion PatchCore 공용 모듈.

파이프라인 (view 1장 = ply 1개 + png 1장):
  a. view ply (m 단위, 약 80만 점) -> FPS 로 n 점 샘플링 (s 번 반복, 반복마다 FPS 시작점 변경)
  b. 샘플 n 점 각각을 중심으로 kNN m 개(자기 자신 포함)를 receptive field point 로 선택 ([4] build_knn 방식)
  c. receptive field point 를 카메라 영상에 투영 ([1] 방식: ×1000 -> M^-1 -> T_sensor_to_cam -> K, dist)
     -> 점 1개당 m 개의 receptive field pixel
  d. 3D : Utonia 인코더 stage 2, 3 출력 (108 + 216 = 324 ch)
     2D : ResNet-101 layer2, layer3 출력 (512 + 1024 = 1536 ch, layer3 은 layer2 해상도로 upsample)
  e. (m, C3) 와 (m, C2) 를 각각 L2 정규화 후 concat -> (C3 + C2, h, w), h*w = m
     -> Conv2d(kernel 1, padding same) 로 (h, w) 는 유지하고 channel 정보만 섞음
     -> (h, w) 평균(또는 flatten) 으로 점별 fusion feature
  f. fusion feature 로 memory bank 구축 / anomaly score 계산 ([2], [3] 코드 재사용)

Fusion Conv2d 는 학습하지 않은 고정 랜덤(orthogonal) 사영이며, memory bank 구축 시 저장한 가중치를
scoring 에서 그대로 불러 사용한다.
"""
import glob
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent              # Manufacutring_Inspection/scripts/2D-3D
PROJECT_DIR = SCRIPT_DIR.parents[1]                       # Manufacutring_Inspection
SCRIPTS_DIR = PROJECT_DIR / "scripts"

DATA_ROOT = PROJECT_DIR / "data" / "2D-3D"               # {BAD|GOOD}_{pid}/{ply,img}, sensor2reference.txt
GOOD_MV_DIR = DATA_ROOT
BAD_MV_DIR = DATA_ROOT
MODEL_DIR = PROJECT_DIR / "model" / "2D-3D"               # MB.npy, MB_source.npy, fusion.pt, config.json, utonia.pth
DEFAULT_OUT = PROJECT_DIR / "outputs" / "2D-3D"
DEFAULT_POSE_ROOT = Path("/mnt/data/Dataset/SiM3D/rubbish_bin")
DEFAULT_UTONIA_CKPT = MODEL_DIR / "utonia.pth"

sys.path.insert(0, str(SCRIPT_DIR))
import utonia_kd  # noqa: E402


def _load_module(name, path):
    """파일 경로로 모듈 import (2D-PatchCore 의 anomaly_score.py 등 이름 충돌 방지)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


coreset_sampling = _load_module("pc_coreset_sampling", SCRIPTS_DIR / "AD" / "coreset_sampling.py")
patchcore_score = _load_module("pc_anomaly_score", SCRIPTS_DIR / "AD" / "anomaly_score.py")


# --------------------------------------------------------------------------- #
# 카메라 ([1] point_cloud_rgb_calibration.py 와 동일한 값)
# --------------------------------------------------------------------------- #
K = np.array([
    [3470.2761957637335, 0.0, 2045.7960174875925],
    [0.0, 3468.896050658857, 1563.6291650560495],
    [0.0, 0.0, 1.0],
])
DIST = np.array([-0.188511844122774, 0.2319794094448776, 0.0007564493888092954,
                 -0.0007526313037295627, -0.22907885958155816])
T_SENSOR_TO_CAM = np.array([
    [9.70302221e-01, 5.27803383e-03, 2.41838258e-01, 2.05147854e-01],
    [8.24782165e-04, -9.99828303e-01, 1.85117098e-02, -8.12195538e+00],
    [2.41894441e-01, -1.77624892e-02, -9.70139976e-01, 4.58130696e+02],
    [0.0, 0.0, 0.0, 1.0],
])


def load_pose(pose_file, view_id):
    """sensor2reference.txt 에서 M{view}_0 (4x4) 읽기 (data_preprocess.ipynb get_matrix_tensor_from_txt 와 동일)."""
    content = Path(pose_file).read_text(encoding="utf-8")
    match = re.search(rf"M{view_id}_0\s*\(\s*\[([^\]]+)\]", content)
    if not match:
        raise ValueError(f"{pose_file} 에서 M{view_id}_0 을 찾을 수 없습니다.")
    nums = [float(x) for x in match.group(1).split(",") if x.strip()]
    return np.array(nums).reshape(4, 4)


def project_points(points, M, img_hw):
    """
    [1] visualize_projection 의 투영 부분.
    points : (N, 3) m 단위
    return : uv (N, 2) float pixel 좌표, depth (N,) 카메라 z(mm), valid (N,) bool (카메라 앞 & 영상 안)
    """
    pts_h = np.hstack([points.astype(np.float64) * 1000.0, np.ones((len(points), 1))])
    moved = (np.linalg.inv(M) @ pts_h.T).T
    cam = (T_SENSOR_TO_CAM @ moved.T).T[:, :3]
    depth = cam[:, 2]
    uv = np.full((len(points), 2), -1.0)
    front = depth > 0
    if front.any():
        img_pts, _ = cv2.projectPoints(cam[front], np.zeros((3, 1)), np.zeros((3, 1)), K, DIST)
        uv[front] = img_pts.reshape(-1, 2)
    h, w = img_hw
    valid = front & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    return uv, depth, valid


# --------------------------------------------------------------------------- #
# 데이터 목록
# --------------------------------------------------------------------------- #
def list_views(state, products=None, views=None, pose_root=DEFAULT_POSE_ROOT):
    """
    state : 'good' | 'bad'
    return: [dict(state, pid, view, stem, ply, img, pose)], (pid, view) 순 정렬
    """
    if state == "good":
        prod_dirs = sorted(p for p in GOOD_MV_DIR.iterdir() if p.is_dir())
    else:
        prod_dirs = sorted(p for p in BAD_MV_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    items = []
    for pdir in prod_dirs:
        pid = pdir.name
        if products and pid not in products:
            continue
        poses = glob.glob(str(pose_root / state / "**" / f"rubbish_bin_{pid}_{state}_sensor2reference.txt"),
                          recursive=True)
        if not poses:
            print(f"[skip] pose 파일 없음: {state} {pid}")
            continue
        for ply in sorted((pdir / "ply").glob(f"rubbish_bin_{pid}_{state}_[0-9][0-9].ply")):
            view = int(ply.stem[-2:])
            if views and view not in views:
                continue
            img = pdir / "img" / f"{ply.stem}_0.png"
            if not img.exists():
                print(f"[skip] 이미지 없음: {img}")
                continue
            items.append(dict(state=state, pid=pid, view=view, stem=ply.stem,
                              ply=ply, img=img, pose=Path(poses[0])))
    return items


def read_ply(path):
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else np.zeros_like(pts)
    return pts, cols


def write_ply(path, points, colors=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(colors, dtype=np.float64), 0, 1))
    o3d.io.write_point_cloud(str(path), pcd)


# --------------------------------------------------------------------------- #
# a. FPS 샘플링 (s 번 반복)
# --------------------------------------------------------------------------- #
def sample_path(out_dir, n, item, k):
    """{out}/{n}_point/{state}/{pid}/{stem}_s{k}.ply"""
    return Path(out_dir) / f"{n}_point" / item["state"] / item["pid"] / f"{item['stem']}_s{k}.ply"


def fps_sample(raw_pts, raw_cols, n, k, seed=0):
    """k 번째 반복 샘플: 반복마다 FPS 시작점을 바꿔 서로 다른 n 점 샘플 생성."""
    start = int(np.random.default_rng(seed + k).integers(len(raw_pts))) if k > 0 else 0
    idx = utonia_kd.fps(raw_pts, n, start_idx=start)
    return raw_pts[idx], raw_cols[idx]


def get_sample(item, raw_pts, raw_cols, n, k, out_dir, resample=False):
    """샘플 ply 가 있으면 재사용, 없으면 FPS 후 저장."""
    path = sample_path(out_dir, n, item, k)
    if path.exists() and not resample:
        return read_ply(path)
    pts, cols = fps_sample(raw_pts, raw_cols, n, k)
    write_ply(path, pts, cols)
    return pts, cols


# --------------------------------------------------------------------------- #
# b. receptive field point (kNN, 자기 자신 포함 m 개)
# --------------------------------------------------------------------------- #
def receptive_field(points, m):
    """return (n, m) int64, 0 번째 열 = 자기 자신, 이후 거리 오름차순."""
    _, idx = cKDTree(points).query(points, k=m, workers=-1)
    return np.asarray(idx, dtype=np.int64).reshape(len(points), m)


def grid_hw(m):
    """m 을 (h, w) 격자로: h = sqrt(m) 이하 최대 약수 (4 -> 2x2, 9 -> 3x3, 6 -> 2x3)."""
    h = max(d for d in range(1, int(np.sqrt(m)) + 1) if m % d == 0)
    return h, m // h


# --------------------------------------------------------------------------- #
# d. feature extractor
# --------------------------------------------------------------------------- #
class Utonia3D:
    """Utonia 인코더 앞쪽 stage 출력 중 선택 stage 만 점별로 반환 (stage 번호 1-based)."""

    def __init__(self, ckpt, stages=(2, 3), scale=6.0, grid_size=0.01, no_flash=False, device="cuda"):
        self.stages = tuple(stages)
        self.model, cfg = utonia_kd.load_teacher(str(ckpt), max(self.stages), no_flash, device)
        ch = list(cfg["enc_channels"])
        offs = np.cumsum([0] + ch)
        self.slices = [slice(int(offs[s - 1]), int(offs[s])) for s in self.stages]
        self.dim = sum(ch[s - 1] for s in self.stages)
        self.transform = utonia_kd.build_transform(scale, grid_size)
        self.device = device

    @torch.no_grad()
    def __call__(self, points, seed=0):
        """points (n, 3) m 단위 -> (n, dim) float32 tensor (GridSample inverse 로 원래 점에 매핑)."""
        pts = points.astype(np.float32)
        nrm = utonia_kd.estimate_normals(pts)
        data = utonia_kd.prepare_input(pts, nrm, self.transform, self.device, seed=seed)
        feat = utonia_kd.point_features(self.model, data)
        feat = torch.cat([feat[:, s] for s in self.slices], dim=1)
        return feat[data["inverse"]].float()


class ResNet2D:
    """ResNet-101 (ImageNet) layer2 + layer3. layer3 은 layer2 해상도로 upsample 후 concat."""

    LAYER_CH = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
    LAYER_ORDER = ("layer1", "layer2", "layer3", "layer4")

    def __init__(self, layers=("layer2", "layer3"), agg=3, device="cuda"):
        import torchvision
        net = torchvision.models.resnet101(weights="IMAGENET1K_V2")
        self.layers = tuple(layers)
        last = max(self.LAYER_ORDER.index(name) for name in self.layers)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool).to(device).eval()
        self.blocks = nn.ModuleList([getattr(net, name) for name in self.LAYER_ORDER[:last + 1]]).to(device).eval()
        self.dim = sum(self.LAYER_CH[name] for name in self.layers)
        self.agg = agg
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, img_rgb):
        """img_rgb (H, W, 3) uint8 -> (1, dim, H/8, W/8) float32 (첫 번째 선택 layer 해상도)."""
        x = torch.from_numpy(img_rgb).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        x = (x - self.mean) / self.std
        outs = {}
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
            x = self.stem(x)
            for name, block in zip(self.LAYER_ORDER, self.blocks):
                x = block(x)
                if name in self.layers:
                    outs[name] = x.float()
        size = outs[self.layers[0]].shape[-2:]
        feats = [F.interpolate(outs[n], size=size, mode="bilinear", align_corners=False)
                 if outs[n].shape[-2:] != size else outs[n] for n in self.layers]
        feat = torch.cat(feats, dim=1)
        if self.agg > 1:  # PatchCore local neighborhood aggregation
            feat = F.avg_pool2d(feat, self.agg, stride=1, padding=self.agg // 2)
        return feat


def crop_box(uv, valid, img_hw, margin=32, align=32):
    """투영된 점들을 감싸는 crop 영역 (align 배수 크기). return x0, y0, x1, y1"""
    h, w = img_hw
    if not valid.any():
        return 0, 0, w, h
    u, v = uv[valid, 0], uv[valid, 1]
    x0, x1 = max(0, int(u.min()) - margin), min(w, int(np.ceil(u.max())) + margin + 1)
    y0, y1 = max(0, int(v.min()) - margin), min(h, int(np.ceil(v.max())) + margin + 1)
    cw = min(w, int(np.ceil((x1 - x0) / align) * align))
    ch = min(h, int(np.ceil((y1 - y0) / align) * align))
    x0, y0 = min(x0, w - cw), min(y0, h - ch)
    return x0, y0, x0 + cw, y0 + ch


def sample_feature_map(fmap, uv, valid, box):
    """
    fmap : (1, C, h, w) crop 영역의 feature map
    uv   : (N, 2) 원본 영상 pixel 좌표
    return (N, C), 영상 밖 / 카메라 뒤 점은 0
    """
    x0, y0, x1, y1 = box
    gx = ((uv[:, 0] - x0) + 0.5) / (x1 - x0) * 2 - 1
    gy = ((uv[:, 1] - y0) + 0.5) / (y1 - y0) * 2 - 1
    grid = torch.from_numpy(np.stack([gx, gy], 1)).float().to(fmap.device).view(1, 1, -1, 2)
    out = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    out = out[0, :, 0].T                                                     # (N, C)
    return out * torch.from_numpy(valid).to(fmap.device).float()[:, None]


# --------------------------------------------------------------------------- #
# e. fusion (Conv2d 로 channel 정보 혼합, (h, w) 유지)
# --------------------------------------------------------------------------- #
class ReceptiveFusion(nn.Module):
    def __init__(self, c3, c2, m, out_dim=256, kernel=1, pool="mean", seed=0):
        super().__init__()
        self.c3, self.c2, self.m = c3, c2, m
        self.h, self.w = grid_hw(m)
        self.pool = pool
        self.out_dim = out_dim
        self.kernel = kernel
        self.conv = nn.Conv2d(c3 + c2, out_dim, kernel_size=kernel, padding=kernel // 2, bias=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            nn.init.orthogonal_(self.conv.weight)   # 거리 보존에 가까운 고정 랜덤 사영 (학습하지 않음)

    @property
    def dim(self):
        return self.out_dim * (self.h * self.w if self.pool == "flatten" else 1)

    def config(self):
        return dict(c3=self.c3, c2=self.c2, m=self.m, out_dim=self.out_dim, kernel=self.kernel, pool=self.pool)

    @torch.no_grad()
    def forward(self, f3, f2):
        """
        f3 : (N, m, C3) receptive field point feature
        f2 : (N, m, C2) receptive field pixel feature
        return (N, dim) 점별 fusion feature
        """
        n = f3.shape[0]
        x = torch.cat([F.normalize(f3, dim=-1), F.normalize(f2, dim=-1)], dim=-1)   # (N, m, C)
        x = x.permute(0, 2, 1).reshape(n, -1, self.h, self.w)                        # (N, C, h, w)
        y = self.conv(x)                                                              # (N, out, h, w)
        return y.mean(dim=(2, 3)) if self.pool == "mean" else y.flatten(1)


def save_fusion(path, fusion):
    torch.save({"config": fusion.config(), "state_dict": fusion.state_dict()}, path)


def load_fusion(path, device):
    ck = torch.load(path, map_location="cpu")
    fusion = ReceptiveFusion(**ck["config"])
    fusion.load_state_dict(ck["state_dict"])
    return fusion.to(device).eval()


# --------------------------------------------------------------------------- #
# 1-a ~ 1-e : view 1개 처리
# --------------------------------------------------------------------------- #
class FusionExtractor:
    def __init__(self, args, device, fusion=None):
        self.args = args
        self.device = device
        self.f3d = Utonia3D(args.utonia_ckpt, args.utonia_stages, args.utonia_scale, args.utonia_grid,
                            args.no_flash, device)
        self.f2d = ResNet2D(args.resnet_layers, args.agg_2d, device)
        self.fusion = fusion if fusion is not None else ReceptiveFusion(
            self.f3d.dim, self.f2d.dim, args.m, args.fusion_dim, args.fusion_kernel, args.fusion_pool,
            args.seed).to(device).eval()
        if (self.fusion.c3, self.fusion.c2, self.fusion.m) != (self.f3d.dim, self.f2d.dim, args.m):
            raise ValueError(f"fusion 설정 불일치: fusion={self.fusion.config()}, "
                             f"현재 c3={self.f3d.dim}, c2={self.f2d.dim}, m={args.m}")

    @torch.no_grad()
    def __call__(self, points, img_rgb, M, seed=0, chunk=8192):
        """
        points  : (n, 3) 샘플 점 (m 단위)
        img_rgb : (H, W, 3) uint8
        return  : dict(feat (n, D) tensor, uv (n, 2), valid (n,) bool, rf (n, m))
        """
        img_hw = img_rgb.shape[:2]
        rf = receptive_field(points, self.args.m)                         # b
        uv, _, valid = project_points(points, M, img_hw)                  # c (receptive field 점 = 샘플 점 집합)
        f3 = self.f3d(points, seed=seed)                                  # d-3D (n, C3)
        box = crop_box(uv, valid, img_hw)
        x0, y0, x1, y1 = box
        fmap = self.f2d(np.ascontiguousarray(img_rgb[y0:y1, x0:x1]))      # d-2D
        f2 = sample_feature_map(fmap, uv, valid, box)                     # (n, C2)
        del fmap
        rf_t = torch.from_numpy(rf).to(self.device)
        feats = []
        for i in range(0, len(points), chunk):                            # e
            idx = rf_t[i:i + chunk]
            feats.append(self.fusion(f3[idx], f2[idx]))
        return dict(feat=torch.cat(feats), uv=uv, valid=valid, rf=rf)


def load_image_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# argparse 공통 인자
# --------------------------------------------------------------------------- #
def add_common_args(p):
    p.add_argument("--n", "--n_points", dest="n", type=int, default=64000, help="view 당 FPS 샘플 점 수")
    p.add_argument("--s", "--n_repeat", dest="s", type=int, default=1, help="동일 입력에 대한 FPS 샘플링 반복 횟수")
    p.add_argument("--m", "--m_pixels", dest="m", type=int, default=4,
                   help="점 1개에 대응되는 receptive field point/pixel 수 (h*w 격자로 배치)")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT, help="output 저장 위치")
    p.add_argument("--tag", default=None, help="memory bank/score 폴더 이름 (기본 n{n}_s{s}_m{m})")
    p.add_argument("--pose_root", type=Path, default=DEFAULT_POSE_ROOT, help="sensor2reference.txt 상위 폴더")
    p.add_argument("--products", nargs="*", default=None, help="일부 제품만 (예: 01 02)")
    p.add_argument("--views", nargs="*", type=int, default=None, help="일부 view 만 (예: 1 4 7)")
    p.add_argument("--resample", action="store_true", help="저장된 샘플 ply 가 있어도 다시 FPS")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def add_model_args(p):
    p.add_argument("--utonia_ckpt", type=Path, default=DEFAULT_UTONIA_CKPT)
    p.add_argument("--utonia_stages", type=int, nargs="+", default=[2, 3], help="3D feature stage (1-based)")
    p.add_argument("--utonia_scale", type=float, default=6.0)
    p.add_argument("--utonia_grid", type=float, default=0.01)
    p.add_argument("--no_flash", action="store_true")
    p.add_argument("--resnet_layers", nargs="+", default=["layer2", "layer3"], help="2D feature layer")
    p.add_argument("--agg_2d", type=int, default=3, help="2D feature local aggregation kernel (1 이면 생략)")
    p.add_argument("--fusion_dim", type=int, default=256, help="fusion Conv2d 출력 channel")
    p.add_argument("--fusion_kernel", type=int, default=1, help="fusion Conv2d kernel (홀수, same padding)")
    p.add_argument("--fusion_pool", choices=["mean", "flatten"], default="mean",
                   help="(h, w) fusion map -> 점별 벡터 방법")
    return p


def resolve_tag(args):
    return args.tag or f"n{args.n}_s{args.s}_m{args.m}"


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
