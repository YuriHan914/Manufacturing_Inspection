"""
4. 대시보드(Summary / Analysis) 용 2D-3D fusion anomaly detection 추론 + 결과 저장.

  object 폴더(data/2D-3D/{BAD|GOOD}_{pid}) 1개 = 12 view (ply/rubbish_bin_{pid}_{state}_{view}.ply
  + img/{stem}_0.png) + rubbish_bin_{pid}_{state}_sensor2reference.txt (카메라 pose).

  view 별 점수는 score_anomaly.py 와 동일 (FPS -> receptive field -> Utonia/ResNet fusion feature
  -> memory bank 1-NN 거리, view score = PatchCore Eq. 7 re-weighting), object score = view score 최댓값.
  GOOD object 는 memory bank 구축에 쓰인 제품이므로 같은 제품 bank feature 를 제외하고 채점(LOO).

  결과 저장은 3D-AD/infer_visualize.py 의 infer_file 방식:
    {out_dir}/{name}/ply/{stem}_score.ply        view 별 score 색 점군 (m, --save_view_ply 일 때만, object 당 ~240MB)
    {out_dir}/{name}/img/{stem}_0_score.jpg      영상 위 score heatmap (visualize_score.score_image, 글자 없음:
                                                 대시보드가 180도 회전해 표시하고 score 는 caption 으로 보여줌)
    {out_dir}/{name}/{name}_score.ply            12 view 병합 object 점군 (mm, voxel downsample)
    {out_dir}/{name}/{name}_scores.npz           points (mm), scores (object 표시용), view_scores
    {out_dir}/{name}/view_scores.csv
    {out_dir}/inference_results.csv              object 별 결과 (대시보드가 읽음)

  판정 ({model_dir}/object_level_thresholds.csv : view 번호별 threshold):
    view   : view_score > 해당 view 의 threshold 이면 abnormal
    object : hard voting - 12 view 중 하나라도 abnormal 이면 Anomaly (bad), 아니면 Normal

  {model_dir}/calibration.json (정상 GOOD LOO 점수 기준):
    point_threshold : 정상 점 score 의 99.5 백분위수 -> anomalous point 비율
    vmin / vmax     : score 색 범위

실행 (프로젝트 루트에서, conda env utonia):
  python scripts/2D-3D/infer_app.py --objects data/2D-3D/BAD_03 data/2D-3D/GOOD_07
"""
import argparse
import json
import re
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

import fusion_common as fc
from visualize_score import interpolate_scores, score_image

MODEL_NAME = "2D-3D AD"
MODEL_KEYS = ("n", "m", "utonia_ckpt", "utonia_stages", "utonia_scale", "utonia_grid", "no_flash",
              "resnet_layers", "agg_2d")
RESULT_COLUMNS = ("model", "path", "name", "state", "pid", "label", "n_abnormal_views", "abnormal_views",
                  "object_score", "point_max", "anomalous_point_ratio", "point_threshold", "n_views", "loo",
                  "inference_ms", "created_at")
OBJECT_DIR_PATTERN = re.compile(r"(good|bad)_(\d+)", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description="2D-3D fusion anomaly detection (dashboard)")
    p.add_argument("--objects", nargs="+", type=Path, required=True, help="object 폴더 (data/2D-3D/BAD_03 ...)")
    p.add_argument("--model_dir", type=Path, default=fc.MODEL_DIR)
    p.add_argument("--out_dir", type=Path, default=fc.DEFAULT_OUT)
    p.add_argument("--pose_root", type=Path, default=fc.DEFAULT_POSE_ROOT,
                   help="object 폴더에 sensor2reference.txt 가 없을 때 찾을 상위 폴더")
    p.add_argument("--s", type=int, default=1, help="FPS 샘플링 반복 횟수")
    p.add_argument("--k_nn", type=int, default=None, help="view score re-weighting neighbor 수 (기본 calibration)")
    p.add_argument("--no_loo", action="store_true", help="GOOD 채점 시 같은 제품 bank feature 도 사용")
    p.add_argument("--save_view_ply", action="store_true", help="view 별 score 색 raw 점군 ply 도 저장")
    p.add_argument("--interp_k", type=int, default=8, help="raw 점 score 보간 kNN 수")
    p.add_argument("--voxel", type=float, default=1.0, help="object 병합 voxel 크기 (mm)")
    p.add_argument("--view_max_points", type=int, default=200000, help="object 표시용 npz 최대 점 수")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--splat", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def relative_key(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(fc.PROJECT_DIR).as_posix()
    except ValueError:
        return resolved.as_posix()


def list_object_views(obj_dir, pose_root):
    """object 폴더 -> (state, pid, [dict(state, pid, view, stem, ply, img, pose)])"""
    obj_dir = Path(obj_dir)
    match = OBJECT_DIR_PATTERN.fullmatch(obj_dir.name)
    if not match:
        raise ValueError(f"object 폴더 이름은 GOOD_<pid> / BAD_<pid> 형식이어야 합니다: {obj_dir}")
    state, pid = match.group(1).lower(), match.group(2)
    pose = obj_dir / f"rubbish_bin_{pid}_{state}_sensor2reference.txt"
    if not pose.exists():
        found = sorted(Path(pose_root).glob(f"{state}/**/{pose.name}"))
        if not found:
            raise FileNotFoundError(f"pose 파일 없음: {pose}")
        pose = found[0]
    items = []
    for ply in sorted((obj_dir / "ply").glob(f"rubbish_bin_{pid}_{state}_[0-9][0-9].ply")):
        img = obj_dir / "img" / f"{ply.stem}_0.png"
        if not img.exists():
            print(f"[skip] 이미지 없음: {img}", flush=True)
            continue
        items.append(dict(state=state, pid=pid, view=int(ply.stem[-2:]), stem=ply.stem, ply=ply, img=img, pose=pose))
    if not items:
        raise FileNotFoundError(f"view(ply + img) 가 없습니다: {obj_dir}")
    return state, pid, items


def voxel_merge(points, scores, voxel):
    """voxel 별 점 좌표 / score 평균 (open3d voxel_down_sample 과 같은 방식, score 도 함께 평균)."""
    grid = np.floor((points - points.min(0)) / voxel).astype(np.int64)
    dims = grid.max(0) + 1
    keys = (grid[:, 0] * dims[1] + grid[:, 1]) * dims[2] + grid[:, 2]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    merged = np.stack([np.bincount(inverse, weights=points[:, i]) for i in range(3)], 1) / counts[:, None]
    merged_scores = np.bincount(inverse, weights=scores) / counts
    return merged.astype(np.float32), merged_scores.astype(np.float32)


def load_view_thresholds(path):
    """object_level_thresholds.csv (view, threshold) -> {view 번호: threshold}"""
    frame = pd.read_csv(path)
    return {int(view): float(thr) for view, thr in zip(frame["view"], frame["threshold"])}


def load_results(path):
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype={"pid": str, "abnormal_views": str}, keep_default_na=False)
    return {str(row["path"]): row.to_dict() for _, row in frame.iterrows()}


def main():
    args = parse_args()
    t_start = time.time()
    model_dir = args.model_dir.resolve()
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    calib = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))
    point_thr = float(calib["point_threshold"])
    view_thr = load_view_thresholds(model_dir / "object_level_thresholds.csv")
    vmin, vmax = float(calib["vmin"]), float(calib["vmax"])
    cmap = matplotlib.colormaps[calib.get("cmap", "jet")]
    k_nn = args.k_nn or int(calib.get("k_nn", 3))

    # 모델 설정은 memory bank 구축 때의 config.json 그대로, utonia_ckpt 는 model_dir 기준
    margs = Namespace(**{key: cfg[key] for key in MODEL_KEYS}, seed=args.seed)
    margs.utonia_ckpt = (model_dir / Path(cfg["utonia_ckpt"]).name).resolve()
    print(f"[model] {model_dir}  (n={margs.n}, m={margs.m}, utonia={margs.utonia_ckpt.name})", flush=True)

    objects = [(obj, *list_object_views(obj, args.pose_root)) for obj in args.objects]
    missing = sorted({item["view"] for *_, items in objects for item in items} - set(view_thr))
    if missing:
        raise ValueError(f"object_level_thresholds.csv 에 view {missing} 의 threshold 가 없습니다.")
    total_views = sum(len(items) for *_, items in objects)

    bank = torch.from_numpy(np.load(model_dir / "MB.npy")).float()
    source = np.load(model_dir / "MB_source.npy")
    fusion = fc.load_fusion(model_dir / "fusion.pt", args.device)
    extractor = fc.FusionExtractor(margs, args.device, fusion=fusion)
    chunk = max(64, int(1.5e8 // len(bank)))
    full_scorer = fc.patchcore_score.PatchCoreScorer(bank, k_nn=k_nn, chunk=chunk)
    print(f"[bank] {tuple(bank.shape)}, 제품 {sorted(set(source.tolist()))}  ({time.time() - t_start:.1f}s)", flush=True)

    results_path = args.out_dir / "inference_results.csv"
    results = load_results(results_path)
    done = 0
    for obj_dir, state, pid, items in objects:
        name = obj_dir.name
        obj_out = args.out_dir / name
        scorer, loo = full_scorer, False
        if state == "good" and not args.no_loo:
            keep = source != int(pid)
            if keep.any() and not keep.all():
                scorer = fc.patchcore_score.PatchCoreScorer(bank[torch.from_numpy(keep)], k_nn=k_nn, chunk=chunk)
                loo = True

        view_rows, obj_pts, obj_scores = [], [], []
        infer_ms = 0.0
        n_points = n_anomalous = 0
        for item in items:
            raw_pts, raw_cols = fc.read_ply(item["ply"])
            img = fc.load_image_rgb(item["img"])
            M = fc.load_pose(item["pose"], item["view"])

            t0 = time.perf_counter()
            pts_all, sc_all, vs_all = [], [], []
            for k in range(args.s):
                pts, _ = fc.fps_sample(raw_pts, raw_cols, margs.n, k)
                feat = extractor(pts, img, M, seed=args.seed + k)["feat"]
                dmin, imin = scorer._nn_distance(feat)
                vs_all.append(float(scorer._image_score(feat, dmin, imin)))
                pts_all.append(pts)
                sc_all.append(dmin.cpu().numpy().astype(np.float32))
            infer_ms += (time.perf_counter() - t0) * 1000.0
            sc = np.stack(sc_all)
            n_points += sc.size
            n_anomalous += int((sc > point_thr).sum())

            # raw 점 score (샘플 점 score 의 역거리 가중 kNN 보간, s 번 평균) -> 색
            raw_scores = np.mean([interpolate_scores(p, s, raw_pts, args.interp_k)
                                  for p, s in zip(pts_all, sc_all)], axis=0)
            colors = cmap(np.clip((raw_scores - vmin) / (vmax - vmin + 1e-12), 0, 1))[:, :3]
            if args.save_view_ply:
                fc.write_ply(obj_out / "ply" / f"{item['stem']}_score.ply", raw_pts, colors)
            obj_pts.append((raw_pts * 1000.0).astype(np.float32))
            obj_scores.append(raw_scores.astype(np.float32))

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            uv, depth, valid = fc.project_points(raw_pts, M, img.shape[:2])
            out_img = score_image(img_bgr, uv, depth, valid, colors, args.alpha, args.splat)
            view_score, threshold = float(np.mean(vs_all)), view_thr[item["view"]]
            abnormal = view_score > threshold
            img_path = obj_out / "img" / f"{item['stem']}_0_score.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            view_rows.append(dict(view=item["view"], stem=item["stem"], view_score=view_score,
                                  threshold=threshold, abnormal=abnormal,
                                  point_max=float(sc.max()), point_mean=float(sc.mean()),
                                  anomalous_point_ratio=float((sc > point_thr).mean() * 100),
                                  image=img_path.relative_to(obj_out).as_posix()))
            done += 1
            print(f"[progress] {done}/{total_views} {item['stem']}  view_score={view_score:.4f} "
                  f"threshold={threshold:.4f}{' ABNORMAL' if abnormal else ''} "
                  f"max={sc.max():.4f}{' (LOO)' if loo else ''}", flush=True)

        views = pd.DataFrame(view_rows)
        views.to_csv(obj_out / "view_scores.csv", index=False)

        # 12 view 병합 object (mm, voxel downsample) -> ply + 표시용 npz
        merged, merged_scores = voxel_merge(np.concatenate(obj_pts), np.concatenate(obj_scores), args.voxel)
        merged_colors = cmap(np.clip((merged_scores - vmin) / (vmax - vmin + 1e-12), 0, 1))[:, :3]
        fc.write_ply(obj_out / f"{name}_score.ply", merged, merged_colors)
        if len(merged) > args.view_max_points:
            sel = np.sort(np.random.default_rng(0).choice(len(merged), args.view_max_points, replace=False))
            merged, merged_scores = merged[sel], merged_scores[sel]
        np.savez(obj_out / f"{name}_scores.npz", points=merged, scores=merged_scores,
                 view_scores=views["view_score"].to_numpy(np.float32))

        # hard voting: view 중 하나라도 abnormal 이면 object 는 Anomaly (bad)
        abnormal_views = views.loc[views["abnormal"], "view"].astype(int).tolist()
        object_score = float(views["view_score"].max())
        key = relative_key(obj_dir)
        results[key] = dict(
            model=MODEL_NAME, path=key, name=name, state=state, pid=pid,
            label="Anomaly" if abnormal_views else "Normal",
            n_abnormal_views=len(abnormal_views), abnormal_views=" ".join(f"{v:02d}" for v in abnormal_views),
            object_score=round(object_score, 4), point_max=round(float(views["point_max"].max()), 4),
            anomalous_point_ratio=round(n_anomalous / max(n_points, 1) * 100, 3),
            point_threshold=round(point_thr, 4),
            n_views=len(views), loo=loo, inference_ms=round(infer_ms, 2),
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        print(f"[object] {name}  abnormal views {len(abnormal_views)}/{len(views)} {abnormal_views} "
              f"-> {results[key]['label']}  "
              f"({len(merged)} 점 표시, {infer_ms / 1000:.1f}s)", flush=True)

        # object 하나 끝날 때마다 저장 (중간에 실패해도 끝난 object 결과는 유지)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([results[k] for k in sorted(results)], columns=list(RESULT_COLUMNS)).to_csv(results_path, index=False)

    print(f"[저장] {args.out_dir}  ({time.time() - t_start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
