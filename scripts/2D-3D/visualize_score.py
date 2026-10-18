"""
3. 점별 anomaly score 를 3D(ply) / 2D(png) 로 표현.

  - view raw ply (약 80만 점) 의 각 점 score = 샘플 점 score 의 역거리 가중 kNN 보간 (s 번 샘플링 평균)
    -> heatmap colormap 으로 점 색 변경 후 ply 저장
  - 12 view ply 를 합쳐 object ply 저장 ([5] 방식: ×1000 (mm) -> 합치기 -> voxel downsample)
  - raw 점을 1-b 와 같은 투영으로 영상에 찍어 (pixel 당 가장 가까운 점, z-buffer) score 색을 영상에 blend 후 png 저장

  색 범위는 모든 채점 결과(GOOD + BAD) 점 score 의 --vmin_pct / --vmax_pct 백분위수 (또는 --vmin / --vmax 지정)
  로 전체 데이터에 동일하게 적용한다.

출력 ({out_dir}/vis/{tag}/{state}/{pid}):
  ply/{stem}_score.ply              view 별 score 색 점군 (원본 단위 m)
  rubbish_bin_{pid}_{state}_score.ply   12 view 를 합친 object (mm, voxel downsample)
  png/{stem}_0_score.png            영상 위 score heatmap
  {out_dir}/vis/{tag}/color_range.json

실행 (2D-3D/ 에서, conda env utonia):
  python Scripts/2D-3D/visualize_score.py --n 64000 --s 1 --m 4
  python Scripts/2D-3D/visualize_score.py --tag smoke --splits bad --products 05 --vmax 0.5
"""
import argparse

import cv2
import matplotlib
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import fusion_common as fc


def parse_args():
    p = argparse.ArgumentParser(description="anomaly score 3D(ply) / 2D(png) 시각화")
    fc.add_common_args(p)
    p.add_argument("--splits", nargs="+", default=["bad", "good"], choices=["bad", "good"])
    p.add_argument("--cmap", default="jet", help="matplotlib colormap")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--vmin_pct", type=float, default=1.0, help="vmin 미지정 시 전체 점 score 백분위수")
    p.add_argument("--vmax_pct", type=float, default=99.9, help="vmax 미지정 시 전체 점 score 백분위수")
    p.add_argument("--interp_k", type=int, default=8, help="raw 점 score 보간 kNN 수")
    p.add_argument("--voxel", type=float, default=1.0, help="object 병합 voxel 크기 (mm, [5] 기본값)")
    p.add_argument("--alpha", type=float, default=0.6, help="png heatmap blend 비율")
    p.add_argument("--splat", type=int, default=1, help="png 에서 점 1개가 칠하는 반경 (pixel, dilate)")
    return p.parse_args()


def interpolate_scores(sample_pts, sample_scores, raw_pts, k):
    """샘플 점 score -> raw 점 score (역거리 가중 kNN)."""
    dist, idx = cKDTree(sample_pts).query(raw_pts, k=k, workers=-1)
    w = 1.0 / (dist + 1e-8)
    return (sample_scores[idx] * w).sum(1) / w.sum(1)


def score_image(img_bgr, uv, depth, valid, colors_rgb, alpha, splat):
    """z-buffer 로 pixel 당 가장 가까운 점의 색을 고른 뒤 영상에 blend."""
    h, w = img_bgr.shape[:2]
    u = np.round(uv[valid, 0]).astype(np.int64).clip(0, w - 1)
    v = np.round(uv[valid, 1]).astype(np.int64).clip(0, h - 1)
    pix = v * w + u
    order = np.lexsort((depth[valid], pix))                   # pixel 별 depth 오름차순
    first = np.ones(len(order), dtype=bool)
    first[1:] = pix[order][1:] != pix[order][:-1]
    sel = order[first]

    color = np.zeros((h * w, 3), dtype=np.uint8)
    mask = np.zeros(h * w, dtype=np.uint8)
    color[pix[sel]] = (colors_rgb[valid][sel][:, ::-1] * 255).astype(np.uint8)   # RGB -> BGR
    mask[pix[sel]] = 255
    color, mask = color.reshape(h, w, 3), mask.reshape(h, w)
    if splat > 0:                                             # 점 사이 빈 pixel 채우기
        kernel = np.ones((2 * splat + 1, 2 * splat + 1), np.uint8)
        color = cv2.dilate(color, kernel)
        mask = cv2.dilate(mask, kernel)
    a = (mask[..., None] / 255.0) * alpha
    return (img_bgr * (1 - a) + color * a).astype(np.uint8)


def main():
    args = parse_args()
    tag = fc.resolve_tag(args)
    score_dir = args.out_dir / "scores" / tag
    vis_dir = args.out_dir / "vis" / tag
    cmap = matplotlib.colormaps[args.cmap]

    jobs = []
    for state in args.splits:
        for item in fc.list_views(state, args.products, args.views, args.pose_root):
            npz = score_dir / state / item["pid"] / f"{item['stem']}.npz"
            if npz.exists():
                jobs.append((item, npz))
            else:
                print(f"[skip] score 없음: {npz}")
    if not jobs:
        raise SystemExit(f"{score_dir} 에 score 파일이 없습니다. score_anomaly.py 를 먼저 실행하세요.")

    all_scores = np.concatenate([np.load(npz)["scores"].ravel() for _, npz in jobs])
    vmin = args.vmin if args.vmin is not None else float(np.percentile(all_scores, args.vmin_pct))
    vmax = args.vmax if args.vmax is not None else float(np.percentile(all_scores, args.vmax_pct))
    fc.save_json(vis_dir / "color_range.json", dict(vmin=vmin, vmax=vmax, cmap=args.cmap,
                                                     vmin_pct=args.vmin_pct, vmax_pct=args.vmax_pct))
    print(f"[color] {args.cmap}  vmin={vmin:.4f}  vmax={vmax:.4f}")

    def save_object(key, pcd):
        state, pid = key
        merged = pcd.voxel_down_sample(voxel_size=args.voxel)
        path = vis_dir / state / pid / f"rubbish_bin_{pid}_{state}_score.ply"
        o3d.io.write_point_cloud(str(path), merged)
        print(f"[object] {path}  ({len(merged.points)} 점)")

    cur_key, cur_obj = None, None          # jobs 는 (state, pid, view) 순 -> 제품이 바뀔 때 object 저장
    for ji, (item, npz) in enumerate(jobs):
        d = np.load(npz)
        raw_pts, _ = fc.read_ply(item["ply"])
        raw_scores = np.mean([interpolate_scores(p, s, raw_pts, args.interp_k)
                              for p, s in zip(d["points"], d["scores"])], axis=0)
        colors = cmap(np.clip((raw_scores - vmin) / (vmax - vmin + 1e-12), 0, 1))[:, :3]

        obj_dir = vis_dir / item["state"] / item["pid"]
        fc.write_ply(obj_dir / "ply" / f"{item['stem']}_score.ply", raw_pts, colors)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raw_pts))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.scale(1000.0, center=(0.0, 0.0, 0.0))
        key = (item["state"], item["pid"])
        if key != cur_key:
            if cur_obj is not None:
                save_object(cur_key, cur_obj)
            cur_key, cur_obj = key, o3d.geometry.PointCloud()
        cur_obj += pcd

        img = cv2.imread(str(item["img"]), cv2.IMREAD_COLOR)
        M = fc.load_pose(item["pose"], item["view"])
        uv, depth, valid = fc.project_points(raw_pts, M, img.shape[:2])
        out_img = score_image(img, uv, depth, valid, colors, args.alpha, args.splat)
        cv2.putText(out_img, f"{item['stem']}  view_score={float(np.mean(d['view_scores'])):.4f}  "
                    f"range=[{vmin:.3f}, {vmax:.3f}]", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                    (255, 255, 255), 5, cv2.LINE_AA)
        png = obj_dir / "png" / f"{item['stem']}_0_score.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(png), out_img)
        print(f"  [{ji + 1}/{len(jobs)}] {item['stem']}  raw {len(raw_pts)} 점, score max {raw_scores.max():.4f}")

    save_object(cur_key, cur_obj)
    print(f"[저장] {vis_dir}")


if __name__ == "__main__":
    main()
