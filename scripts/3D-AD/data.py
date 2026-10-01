"""
데이터 유틸리티
- .off (ModelNet10) / .stl 메시 로드
- 면적 가중 표면 샘플링 + Farthest Point Sampling(FPS)
- kNN 그래프 (scipy cKDTree)
- ModelNet10 합성 장면 생성 (논문 4.1절)
- 식 (4) 스케일 계수 s
- 전처리 결과 캐시 (.npz: points, knn)
"""
import glob
import os
import random
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

try:
    import fpsample
except ImportError:
    fpsample = None


# --------------------------------------------------------------------------- #
# 메시 로드
# --------------------------------------------------------------------------- #
def load_off(path):
    """
    ModelNet .off 로더. 일부 파일은 헤더가 'OFF490 518 0' 처럼 한 줄에 붙어 있어
    일반 로더가 실패하므로 직접 파싱한다.
    return: vertices (V,3) float64, faces (F,3) int64 (다각형은 fan 삼각분할)
    """
    with open(path, "r") as f:
        tokens = f.read().split()
    t0 = tokens[0]
    if t0 == "OFF":
        i = 1
    elif t0.startswith("OFF"):
        tokens = [t0[3:]] + tokens[1:]
        i = 0
    else:
        raise ValueError(f"not an OFF file: {path}")
    nv, nf = int(tokens[i]), int(tokens[i + 1])
    i += 3
    verts = np.array(tokens[i:i + 3 * nv], dtype=np.float64).reshape(nv, 3)
    i += 3 * nv
    faces = []
    for _ in range(nf):
        c = int(tokens[i])
        idx = [int(x) for x in tokens[i + 1:i + 1 + c]]
        i += 1 + c
        for j in range(1, c - 1):
            faces.append((idx[0], idx[j], idx[j + 1]))
    return verts, np.asarray(faces, dtype=np.int64)


def load_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".off":
        return load_off(path)
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 샘플링
# --------------------------------------------------------------------------- #
def sample_surface(verts, faces, num, rng):
    """면적 가중 균일 표면 샘플링."""
    tri = verts[faces]                                   # (F,3,3)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    prob = area / area.sum()
    fid = rng.choice(len(faces), size=num, p=prob)
    u, v = rng.random(num), rng.random(num)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[fid]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def farthest_point_sampling(points, n, start_idx=None):
    """FPS. fpsample(빠름)이 있으면 사용, 없으면 numpy 구현. start_idx=None 이면 무작위 시작점."""
    if len(points) <= n:
        return points
    if fpsample is not None:
        idx = fpsample.bucket_fps_kdline_sampling(points.astype(np.float32), n, h=7, start_idx=start_idx)
        return points[np.asarray(idx)]
    idx = np.zeros(n, dtype=np.int64)
    dist = np.full(len(points), np.inf)
    idx[0] = np.random.randint(len(points)) if start_idx is None else start_idx
    for i in range(1, n):
        dist = np.minimum(dist, ((points - points[idx[i - 1]]) ** 2).sum(1))
        idx[i] = int(dist.argmax())
    return points[idx]


def mesh_to_points(verts, faces, n, rng, oversample=4, start_idx=None):
    """메시 표면을 조밀하게 샘플링한 뒤 FPS로 n개 선택."""
    dense = sample_surface(verts, faces, n * oversample, rng)
    return farthest_point_sampling(dense, n, start_idx).astype(np.float32)


def load_geometry(path):
    """메시 또는 점군 로드. return: vertices (V,3), faces (F,3) 또는 점군이면 None."""
    if os.path.splitext(path)[1].lower() == ".off":
        return load_off(path)
    import trimesh
    m = trimesh.load(path, process=False)
    f = getattr(m, "faces", None)
    f = np.asarray(f, dtype=np.int64) if f is not None and len(f) else None
    return np.asarray(m.vertices, dtype=np.float64), f


def sample_points(verts, faces, n, rng, start_idx=None):
    """면이 있으면 표면 샘플링 + FPS, 점군이면 정점에서 바로 FPS."""
    if faces is None:
        return farthest_point_sampling(verts, n, start_idx).astype(np.float32)
    return mesh_to_points(verts, faces, n, rng, start_idx=start_idx)


def load_points(path, n, rng):
    """메시 또는 점군 파일 -> n개 점."""
    return sample_points(*load_geometry(path), n, rng)


def build_knn(points, k):
    """자기 자신을 제외한 k-최근접 이웃. return (n,k) int64, (n,k) 거리."""
    tree = cKDTree(points)
    dist, idx = tree.query(points, k=k + 1, workers=-1)
    return idx[:, 1:].astype(np.int64), dist[:, 1:]


# --------------------------------------------------------------------------- #
# 식 (4): s = 1/(k|P|) Σ_p Σ_{q∈knn(p)} ||p - q||  (데이터셋 전체 평균)
# --------------------------------------------------------------------------- #
def compute_scale(cache_files, k):
    total, count = 0.0, 0
    for f in cache_files:
        d = np.load(f)
        pts, knn = d["points"], d["knn"][:, :k]
        dist = np.linalg.norm(pts[:, None, :] - pts[knn], axis=-1)
        total += dist.sum()
        count += dist.size
    return float(total / count)


# --------------------------------------------------------------------------- #
# ModelNet10 합성 장면 (논문 4.1절)
# --------------------------------------------------------------------------- #
def random_rotation(rng):
    ax, ay, az = rng.uniform(0, 2 * np.pi, 3)
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_scene(off_files, rng, n, objects_per_scene=10, extent=3.0):
    """
    ModelNet10에서 10개 객체 무작위 선택
    -> 바운딩박스 최장변 = 1 로 스케일
    -> 각 축 [0, 2π] 균등 회전
    -> [-3, 3]^3 균등 위치
    -> 장면 전체에서 FPS로 n점
    """
    all_v, all_f, offset = [], [], 0
    for path in rng.choice(off_files, size=objects_per_scene, replace=False):
        v, f = load_off(path)
        v = v - (v.max(0) + v.min(0)) / 2
        v = v / (v.max(0) - v.min(0)).max()
        v = v @ random_rotation(rng).T + rng.uniform(-extent, extent, 3)
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)
    return mesh_to_points(np.concatenate(all_v), np.concatenate(all_f), n, rng)


def find_modelnet_off(root, split="train"):
    files = sorted(glob.glob(os.path.join(root, "*", split, "*.off")))
    if not files:
        files = sorted(glob.glob(os.path.join(root, "**", "*.off"), recursive=True))
    return files


def load_modelnet_hf(repo_id="naderalfares/ModelNet10", cache_dir=None, max_workers=2):
    """
    HuggingFace Hub에서 ModelNet10 다운로드.
    load_dataset은 메타데이터(object_path 문자열)만 제공하므로, 실제 .off 메시는
    snapshot_download로 받아 로컬 경로로 매핑한다.
    파일이 ~4.9k개라 HF 요청 한도(429)에 걸리기 쉬우므로 동시 다운로드 수를 낮게 둔다.
    중단되어도 다시 실행하면 캐시된 파일은 건너뛴다.
    return: {"train": [.off 경로...], "test": [...]}
    """
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    ds = load_dataset(repo_id, cache_dir=cache_dir)
    local = snapshot_download(repo_id, repo_type="dataset", cache_dir=cache_dir,
                              allow_patterns=["ModelNet10/**/*.off"], max_workers=max_workers)
    root = os.path.join(local, "ModelNet10")
    return {split: sorted(os.path.join(root, p) for p in ds[split]["object_path"])
            for split in ds}


def find_meshes(root):
    exts = ("*.stl", "*.STL", "*.off", "*.obj", "*.ply")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, "**", e), recursive=True)
    return sorted(set(files))


def save_cache(path, points, k):
    knn, _ = build_knn(points, k)
    np.savez(path, points=points.astype(np.float32), knn=knn.astype(np.int32))


def load_cache(path):
    d = np.load(path)
    return d["points"].astype(np.float32), d["knn"].astype(np.int64)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for st in self.streams:
            st.write(data)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def setup_log(out_dir, name="train.log"):
    """stdout/stderr를 화면과 out_dir/train.log에 동시에 기록 (이어쓰기)."""
    f = open(os.path.join(out_dir, name), "a", encoding="utf-8")
    f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} | {' '.join(sys.argv)} =====\n")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    return f


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
