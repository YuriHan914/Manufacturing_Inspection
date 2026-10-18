"""
Utonia(PTv3) 인코더 로드 / 입력 전처리 공용 모듈 (2D-3D fusion 의 3D feature 용).

원본: Anomaly-Detection/2D-3D/Scripts/3D-Knowledge-Distillation/utonia/utonia_kd.py 에서
fusion_common.py 가 쓰는 부분만 가져옴 (teacher 로드, 점별 multi-scale feature, FPS, 법선, transform).

utonia 패키지(thirdparty/Utonia)가 설치된 환경(conda env utonia)에서 실행한다.
"""
import os

import numpy as np
import torch

import utonia  # noqa: E402
from utonia.model import PointTransformerV3  # noqa: E402

try:
    import fpsample
except ImportError:
    fpsample = None

UTONIA_REPO_ID = "Pointcept/Utonia"
ENC_KEYS = ("enc_depths", "enc_channels", "enc_num_head", "enc_patch_size")

# 체크포인트에 config 가 없을 때 (Pointcept 학습 체크포인트) 사용하는 Utonia 구조
UTONIA_CONFIG = dict(
    in_channels=9, order=("z", "z-trans", "hilbert", "hilbert-trans"), stride=(2, 2, 2, 2),
    enc_depths=(3, 3, 3, 12, 3), enc_channels=(54, 108, 216, 432, 576), enc_num_head=(3, 6, 12, 24, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024), mlp_ratio=4, qkv_bias=True, qk_scale=None,
    attn_drop=0.0, proj_drop=0.0, drop_path=0.3, shuffle_orders=True, pre_norm=True, enable_rpe=False,
    enable_flash=True, upcast_attention=False, upcast_softmax=False, traceable=True, enc_mode=True,
    mask_token=True, rope_base=10,
)


# --------------------------------------------------------------------------- #
# 모델
# --------------------------------------------------------------------------- #
def ensure_utonia_ckpt(ckpt):
    """ckpt 파일이 없으면 HuggingFace(Pointcept/Utonia) 에서 같은 이름으로 받아 ckpt 위치에 저장."""
    ckpt = os.path.abspath(str(ckpt))
    if os.path.isfile(ckpt):
        return ckpt
    from huggingface_hub import hf_hub_download
    print(f"[utonia] {ckpt} 없음 -> HuggingFace {UTONIA_REPO_ID}/{os.path.basename(ckpt)} 다운로드", flush=True)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    return hf_hub_download(repo_id=UTONIA_REPO_ID, filename=os.path.basename(ckpt),
                           local_dir=os.path.dirname(ckpt))


def load_utonia_ckpt(ckpt):
    """
    Utonia 가중치 로드. return: config, backbone state_dict
    ('module.' / 'backbone.' 접두사를 떼어 PTv3 backbone 키로 맞춤)
    """
    ck = torch.load(ensure_utonia_ckpt(ckpt), map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck)
    out = {}
    for key, value in state.items():
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        out[key] = value
    return dict(ck.get("config", UTONIA_CONFIG)), out


def utonia_config(ckpt, num_stages=3, no_flash=False):
    """Utonia 체크포인트 로드 후 인코더를 앞쪽 num_stages 개 stage 로 자름."""
    cfg, state = load_utonia_ckpt(ckpt)
    for key in ENC_KEYS:
        cfg[key] = tuple(cfg[key][:num_stages])
    cfg["stride"] = tuple(cfg["stride"][:num_stages - 1])
    cfg.update(enc_mode=True, traceable=True)
    if no_flash:
        cfg.update(enable_flash=False, enc_patch_size=tuple([1024] * num_stages))
    return cfg, state


def load_teacher(ckpt, num_stages, no_flash, device):
    cfg, state = utonia_config(ckpt, num_stages, no_flash)
    teacher = PointTransformerV3(**cfg)
    missing = teacher.load_state_dict(state, strict=False).missing_keys
    if missing:
        raise RuntimeError(f"teacher 가중치 누락: {missing[:5]} ...")
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, cfg


def point_features(model, data):
    """
    data   : prepare_input() 결과 (device 위 텐서)
    return : (n_grid, Σ enc_channels)  stage-0 격자 점별 multi-scale 특징
    """
    point = model(dict(data))
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point.feat


# --------------------------------------------------------------------------- #
# 데이터
# --------------------------------------------------------------------------- #
def fps(points, n, start_idx=0):
    """FPS 인덱스. start_idx 를 바꾸면 같은 파일에서 다른 샘플이 나옴."""
    if len(points) <= n:
        return np.arange(len(points))
    if fpsample is not None:
        return np.asarray(fpsample.bucket_fps_kdline_sampling(
            points.astype(np.float32), n, h=7, start_idx=int(start_idx)), dtype=np.int64)
    idx = np.zeros(n, dtype=np.int64)
    dist = np.full(len(points), np.inf)
    idx[0] = start_idx
    for i in range(1, n):
        dist = np.minimum(dist, ((points - points[idx[i - 1]]) ** 2).sum(1))
        idx[i] = int(dist.argmax())
    return idx


def estimate_normals(points, k=30):
    """kNN 법선 + 무게중심 바깥쪽으로 부호 통일 (학습/추론 동일 규칙)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(k))
    nrm = np.asarray(pcd.normals, dtype=np.float32)
    flip = ((points - points.mean(0)) * nrm).sum(1) < 0
    nrm[flip] *= -1
    return nrm


def build_transform(scale, grid_size):
    return utonia.transform.Compose([
        dict(type="RandomScale", scale=[scale, scale]),
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train",
             return_grid_coord=True, return_inverse=True),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse"),
             feat_keys=("coord", "color", "normal")),
    ])


def prepare_input(points, normals, transform, device, seed=None):
    """
    points, normals : (n, 3)
    seed            : GridSample 복셀 대표점(np.random) + 직렬화 순서 셔플(torch) 고정용.
                      None 이면 무작위 (학습 증강)
    return          : Utonia 입력 dict, data["inverse"] : (n,) 원래 점 -> 격자 점 인덱스
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    data = transform({"coord": points.copy(), "normal": normals.copy(),
                      "color": np.zeros_like(points)})
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in data.items()}
