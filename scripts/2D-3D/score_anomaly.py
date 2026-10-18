"""
2. BAD / GOOD multi-view 데이터의 점별 anomaly score 계산.

  view 마다 build_memory_bank.py 와 같은 과정(FPS -> receptive field -> 3D/2D feature -> fusion)으로
  점별 fusion feature 를 만든 뒤, memory bank 까지의 1-NN 거리를 점 score 로 사용 ([3] PatchCoreScorer).
  view score 는 [3] 의 softmax re-weighting (PatchCore Eq. 7), object score 는 12 view score 의 최댓값.

  GOOD 은 memory bank 구축에 사용된 데이터이므로, 기본적으로 같은 제품에서 나온 bank feature 를 제외하고
  채점한다 (leave-one-product-out, --no_loo 로 끔).

  모델 설정(n, m, utonia/resnet layer 등)은 memory_bank/{tag}/config.json 을 그대로 사용한다.

출력 ({out_dir}/scores/{tag}):
  {state}/{pid}/{stem}.npz      points (s, n, 3), scores (s, n), view_scores (s,)
  view_scores.csv, object_scores.csv, metrics.json, score.log

실행 (2D-3D/ 에서, conda env utonia):
  python Scripts/2D-3D/score_anomaly.py --n 64000 --s 1 --m 4
  python Scripts/2D-3D/score_anomaly.py --tag smoke --splits bad --products 05
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import fusion_common as fc
from data import setup_log

MODEL_KEYS = ("n", "m", "utonia_ckpt", "utonia_stages", "utonia_scale", "utonia_grid", "no_flash",
              "resnet_layers", "agg_2d")


def parse_args():
    p = argparse.ArgumentParser(description="memory bank 기반 2D-3D fusion anomaly score")
    fc.add_common_args(p)
    p.add_argument("--splits", nargs="+", default=["bad", "good"], choices=["bad", "good"])
    p.add_argument("--k_nn", type=int, default=3, help="view score re-weighting neighbor 수 ([3] Eq. 7)")
    p.add_argument("--no_loo", action="store_true", help="GOOD 채점 시 같은 제품 bank feature 도 사용")
    return p.parse_args()


def auroc(labels, scores):
    """Mann-Whitney U 기반 AUROC (동점은 평균 순위)."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = pd.Series(scores).rank().to_numpy()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def main():
    args = parse_args()
    tag = fc.resolve_tag(args)
    bank_dir = args.out_dir / "memory_bank" / tag
    score_dir = args.out_dir / "scores" / tag
    score_dir.mkdir(parents=True, exist_ok=True)
    setup_log(str(score_dir), "score.log")
    t0 = time.time()

    cfg = json.loads((bank_dir / "config.json").read_text(encoding="utf-8"))
    for key in MODEL_KEYS:
        setattr(args, key, cfg[key])
    args.utonia_ckpt = fc.Path(args.utonia_ckpt)
    print(f"[bank] {bank_dir}  (n={args.n}, s={args.s}, m={args.m})")

    bank = torch.from_numpy(np.load(bank_dir / "MB.npy")).float()
    source = np.load(bank_dir / "MB_source.npy")
    fusion = fc.load_fusion(bank_dir / "fusion.pt", args.device)
    extractor = fc.FusionExtractor(args, args.device, fusion=fusion)
    chunk = max(64, int(1.5e8 // len(bank)))
    full_scorer = fc.patchcore_score.PatchCoreScorer(bank, k_nn=args.k_nn, chunk=chunk)
    print(f"[bank] {tuple(bank.shape)}, 제품 {sorted(set(source.tolist()))}")

    rows = []
    for state in args.splits:
        items = fc.list_views(state, args.products, args.views, args.pose_root)
        print(f"[{state}] views: {len(items)}")
        scorers = {}
        for vi, item in enumerate(items):
            tv = time.time()
            pid = int(item["pid"])
            scorer, loo = full_scorer, False
            if state == "good" and not args.no_loo:
                keep = source != pid
                if keep.any() and not keep.all():
                    if pid not in scorers:
                        scorers[pid] = fc.patchcore_score.PatchCoreScorer(bank[torch.from_numpy(keep)],
                                                                          k_nn=args.k_nn, chunk=chunk)
                    scorer, loo = scorers[pid], True
                elif not keep.any():
                    print(f"  [warn] bank 가 제품 {item['pid']} 만으로 구성 -> LOO 불가, 전체 bank 사용")

            raw_pts, raw_cols = fc.read_ply(item["ply"])
            img = fc.load_image_rgb(item["img"])
            M = fc.load_pose(item["pose"], item["view"])
            pts_all, sc_all, vs_all = [], [], []
            for k in range(args.s):
                pts, _ = fc.get_sample(item, raw_pts, raw_cols, args.n, k, args.out_dir, args.resample)
                feat = extractor(pts, img, M, seed=args.seed + k)["feat"]
                dmin, imin = scorer._nn_distance(feat)
                vs = scorer._image_score(feat, dmin, imin)
                pts_all.append(pts.astype(np.float32))
                sc_all.append(dmin.cpu().numpy().astype(np.float32))
                vs_all.append(float(vs))

            out = score_dir / state / item["pid"] / f"{item['stem']}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, points=np.stack(pts_all), scores=np.stack(sc_all), view_scores=np.array(vs_all))
            sc = np.stack(sc_all)
            rows.append(dict(state=state, label=int(state == "bad"), pid=item["pid"], view=item["view"],
                             stem=item["stem"], view_score=float(np.mean(vs_all)), point_max=float(sc.max()),
                             point_mean=float(sc.mean()), point_p99=float(np.percentile(sc, 99)), loo=loo))
            print(f"  [{vi + 1}/{len(items)}] {item['stem']}  view_score={np.mean(vs_all):.4f} "
                  f"max={sc.max():.4f} mean={sc.mean():.4f}{' (LOO)' if loo else ''}  {time.time() - tv:.1f}s")

    views = pd.DataFrame(rows)
    views.to_csv(score_dir / "view_scores.csv", index=False)
    objects = (views.groupby(["state", "label", "pid"], as_index=False)
               .agg(object_score=("view_score", "max"), point_max=("point_max", "max"), n_views=("view", "count")))
    objects.to_csv(score_dir / "object_scores.csv", index=False)
    metrics = dict(tag=tag, view_auroc=auroc(views["label"], views["view_score"]),
                   object_auroc=auroc(objects["label"], objects["object_score"]),
                   object_auroc_point_max=auroc(objects["label"], objects["point_max"]),
                   n_views=len(views), n_objects=len(objects), elapsed_sec=round(time.time() - t0, 1))
    fc.save_json(score_dir / "metrics.json", metrics)
    print(objects.to_string(index=False))
    print(f"[metrics] {metrics}")
    print(f"[저장] {score_dir}")


if __name__ == "__main__":
    main()
