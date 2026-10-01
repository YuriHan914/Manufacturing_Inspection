"""
[3단계] 이상 데이터(.ply 점군 / .stl 메시)의 점별 이상 점수 계산 및 3D 시각화

점수: ||f_S - (f_T - μ)/σ||_2  (점별 회귀 오차)
임계값: 정상 검증셋 점수의 백분위수 (기본 99.5%)

출력(파일마다):
  <name>_points.ply      이상 점수 색상 포인트 클라우드 (MeshLab/CloudCompare/Open3D)
  <name>_mesh.ply        원본 메시 정점에 점수 보간 후 색상 입힌 메시 (입력이 메시일 때)
  <name>_full.ply        원본 전체 점에 점수 보간 후 색상 입힌 점군 (입력이 점군일 때)
  <name>_scores.npz      points, scores, mask
  <name>.html            Plotly 인터랙티브 3D 뷰 (브라우저) — 현재 생성 비활성화 (infer_file 참고)

예시:
python infer_visualize.py --student ./runs/student/student_best.pt --out_dir ./runs/vis
# 이상 데이터가 mm, 학습 데이터가 m 인 경우
python infer_visualize.py --student ./runs/student/student_best.pt --out_dir ./runs/vis --input_scale 0.001
"""
import argparse
import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from data import build_knn, find_meshes, load_cache, load_geometry, sample_points
from model import FeatureExtractor, anomaly_scores

DEFAULT_ANOMALY_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../Data/point_cloud/good/3.good_transform_crop_inverse/ply/whole"))


def colormap(x):
    """0~1 값을 jet 계열 RGB(0~255)로."""
    try:
        import matplotlib
        return (matplotlib.colormaps["jet"](x)[:, :3] * 255).astype(np.uint8)
    except Exception:
        r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
        return (np.stack([r, g, b], 1) * 255).astype(np.uint8)


def write_ply(path, verts, colors, faces=None):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if faces is not None:
            f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(verts, colors):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
        if faces is not None:
            for t in faces:
                f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def write_html(path, pts, scores, thr, title, max_points=60000):
    import plotly.graph_objects as go
    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, scores = pts[sel], scores[sel]
    fig = go.Figure(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
        marker=dict(size=1.5, color=scores, colorscale="Jet", cmin=0, cmax=max(thr * 1.5, 1e-6),
                    colorbar=dict(title="anomaly score")),
        hovertemplate="score %{marker.color:.3f}<extra></extra>"))
    fig.update_layout(title=f"{title}  (threshold={thr:.3f})",
                      scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(path, include_plotlyjs="cdn")


class KDScorer:
    """Teacher/student 3D-KD 모델 로드 + 점별 이상 점수 계산."""

    def __init__(self, student_path, teacher_path=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(student_path, map_location=self.device, weights_only=False)
        # 체크포인트에 기록된 teacher 경로가 없으면 student 옆의 teacher_best.pt 사용
        if teacher_path is None:
            teacher_path = ck["teacher_path"]
            if not os.path.exists(teacher_path):
                teacher_path = os.path.join(os.path.dirname(os.path.abspath(student_path)), "teacher_best.pt")
        tck = torch.load(teacher_path, map_location=self.device, weights_only=False)

        self.teacher = FeatureExtractor(ck["d"], ck["num_blocks"]).to(self.device).eval()
        self.teacher.load_state_dict(tck["state_dict"])
        self.student = FeatureExtractor(ck["d"], ck["num_blocks"]).to(self.device).eval()
        self.student.load_state_dict(ck["student"])
        self.mu, self.sigma = ck["mu"].to(self.device), ck["sigma"].to(self.device)
        self.s, self.k, self.n = ck["scale"], ck["k"], ck["n_points"]
        self.val_files = [f for f in ck.get("val_files", []) if os.path.exists(f)]

    @torch.no_grad()
    def score(self, pts, knn):
        p = torch.from_numpy(pts / self.s).to(self.device)
        kn = torch.from_numpy(knn).to(self.device)
        return anomaly_scores(self.student(p, kn), self.teacher(p, kn), self.mu, self.sigma).cpu().numpy()

    def prepare(self, path, input_scale=1.0, auto_scale=False, seed=0):
        """파일 -> (원본 정점, 면, 샘플 점, knn, 적용된 input_scale)."""
        v0, f = load_geometry(path)
        for _ in range(2):
            v = v0 * input_scale
            # FPS 시작점 고정: 같은 파일이면 항상 같은 점/점수 (임계값 계산과 추론 결과 일치)
            pts = sample_points(v, f, self.n, np.random.default_rng(seed), start_idx=0)
            knn, kd = build_knn(pts, self.k)
            # 단위 검사: 학습셋은 평균 kNN 거리 / s ≈ 1 이 되도록 정규화되어 있음
            ratio = kd.mean() / self.s
            if 1 / 3 < ratio < 3:
                break
            hint = 10.0 ** round(-np.log10(ratio)) * input_scale
            if not auto_scale:
                print(f"warning: {os.path.basename(path)} 평균 kNN 거리가 학습 데이터의 {ratio:.3g}배 -> 단위가 다를 수 있음 "
                      f"(--input_scale {hint:g} 시도)")
                break
            input_scale = hint
        return v, f, pts, knn, input_scale

    def file_scores(self, path, input_scale=1.0, auto_scale=False, seed=0):
        _v, _f, pts, knn, _scale = self.prepare(path, input_scale, auto_scale, seed)
        return self.score(pts, knn)


def compute_thresholds(scorer, percentile=99.5, normal_files=None, input_scale=1.0, auto_scale=False):
    """
    point_thr: 정상 점별 점수의 백분위수 (체크포인트 val_files 우선, 없으면 normal_files)
    image_thr: 정상 샘플별 max 점수의 최댓값 (image-level 판정용)
    """
    if scorer.val_files:
        per_file = [scorer.score(*load_cache(f)) for f in scorer.val_files]
        source = "val_files"
    elif normal_files:
        per_file = [scorer.file_scores(f, input_scale, auto_scale) for f in normal_files]
        source = "normal_files"
    else:
        raise RuntimeError("정상 기준 데이터가 없습니다: 체크포인트 val_files 경로가 없고 normal_files 도 비어 있음")
    normal = np.concatenate(per_file)
    point_thr = float(np.percentile(normal, percentile))
    image_thr = float(max(sc.max() for sc in per_file))
    print(f"normal scores ({source}, {len(per_file)} files): mean {normal.mean():.3f}, "
          f"p{percentile} = {point_thr:.3f}, max = {image_thr:.3f}")
    return point_thr, image_thr, source


def infer_file(scorer, path, out_dir, point_thr, input_scale=1.0, auto_scale=False, show=False):
    """파일 하나 추론 + 시각화 결과 저장. return: 결과 dict."""
    name = os.path.splitext(os.path.basename(path))[0]
    t0 = time.perf_counter()
    v, f, pts, knn, used_scale = scorer.prepare(path, input_scale, auto_scale)
    sc = scorer.score(pts, knn)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    mask = sc > point_thr

    col = colormap(np.clip(sc / (point_thr * 1.5), 0, 1))
    write_ply(os.path.join(out_dir, f"{name}_points.ply"), pts, col)

    # 원본 정점(메시) / 전체 점(점군)으로 점수 보간 (역거리 가중, 8-NN)
    dist, idx = cKDTree(pts).query(v, k=8, workers=-1)
    w = 1.0 / (dist + 1e-9)
    vs = (sc[idx] * w).sum(1) / w.sum(1)
    write_ply(os.path.join(out_dir, f"{name}_{'full' if f is None else 'mesh'}.ply"), v,
              colormap(np.clip(vs / (point_thr * 1.5), 0, 1)), f)

    np.savez(os.path.join(out_dir, f"{name}_scores.npz"), points=pts, scores=sc, mask=mask,
             full_scores=vs.astype(np.float32))
    # HTML 뷰 생성 비활성화 (앱은 _full.ply + _scores.npz 로 직접 3D 표시)
    # write_html(os.path.join(out_dir, f"{name}.html"), pts, sc, point_thr, name)
    print(f"{name:30s} max score {sc.max():.3f} | anomalous points {mask.mean() * 100:.2f}%")

    if show:
        import open3d as o3d
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pc.colors = o3d.utility.Vector3dVector(col / 255.0)
        o3d.visualization.draw_geometries([pc], window_name=name)

    return {
        "name": name,
        "path": os.path.abspath(path),
        "max_score": float(sc.max()),
        "anomalous_point_ratio": float(mask.mean() * 100),
        "input_scale": float(used_scale),
        "inference_ms": infer_ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", default=None, help="기본: student 체크포인트에 기록된 경로")
    ap.add_argument("--anomaly_dir", default=DEFAULT_ANOMALY_DIR, help="이상 .ply/.stl 폴더")
    ap.add_argument("--out_dir", default="./runs/vis")
    ap.add_argument("--percentile", type=float, default=99.5,
                    help="정상 검증셋 점별 점수의 이 백분위수를 임계값으로 사용")
    ap.add_argument("--normal_files", nargs="*", default=None,
                    help="체크포인트의 val_files 를 찾을 수 없을 때 임계값 계산에 쓸 정상 파일")
    ap.add_argument("--input_scale", type=float, default=1.0,
                    help="이상 데이터 좌표에 곱할 배율 (학습 데이터와 단위 맞춤, 예: mm->m 이면 0.001)")
    ap.add_argument("--auto_scale", action="store_true", help="kNN 거리 검사로 input_scale 자동 보정")
    ap.add_argument("--show", action="store_true", help="Open3D 창으로 바로 보기")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scorer = KDScorer(args.student, args.teacher, args.device)

    # ---- 임계값: 정상 검증셋 점수 분포 ----
    thr, _image_thr, _source = compute_thresholds(
        scorer, args.percentile, args.normal_files, args.input_scale, args.auto_scale)

    rows = [infer_file(scorer, path, args.out_dir, thr, args.input_scale, args.auto_scale, args.show)
            for path in find_meshes(args.anomaly_dir)]

    with open(os.path.join(args.out_dir, "summary.csv"), "w") as fcsv:
        fcsv.write("name,max_score(image-level),anomalous_point_ratio(%)\n")
        for r in rows:
            fcsv.write(f"{r['name']},{r['max_score']:.4f},{r['anomalous_point_ratio']:.3f}\n")
    print(f"results -> {args.out_dir}")


if __name__ == "__main__":
    main()
