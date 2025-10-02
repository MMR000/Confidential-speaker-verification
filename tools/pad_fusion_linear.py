#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将两个 scores TSV（score\tlabel(optional)\tpath）按 path 对齐：
- z-norm 两边分数
- 线性融合：s = w1*z1 + w2*z2  （默认0.5/0.5）
- 输出同格式 TSV
"""
import os, sys, argparse, csv
import numpy as np

def read_scores(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            if not line.strip(): continue
            parts = [p.strip() for p in line.strip().split("\t")]
            s = float(parts[0])
            lab = parts[1] if len(parts) >= 3 else ""
            pth = parts[-1]
            d[pth] = (s, lab)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores1", required=True, help="比如 CAMPPlus 的 scores_pad.txt")
    ap.add_argument("--scores2", required=True, help="比如 AASIST3 的 scores_aasist3.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w1", type=float, default=0.5)
    args = ap.parse_args()
    w1, w2 = args.w1, 1.0 - args.w1

    s1 = read_scores(args.scores1)
    s2 = read_scores(args.scores2)
    inter = sorted(set(s1.keys()) & set(s2.keys()))
    if not inter:
        print("[ERR] 两个分数文件没有交集 path，无法融合")
        sys.exit(2)

    v1 = np.array([s1[p][0] for p in inter], dtype=float)
    v2 = np.array([s2[p][0] for p in inter], dtype=float)
    # z-norm
    def znorm(x):
        mu = x.mean(); sd = x.std() if x.std() > 1e-8 else 1.0
        return (x - mu) / sd
    z1, z2 = znorm(v1), znorm(v2)
    vf = w1 * z1 + w2 * z2

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["score", "label(optional)", "path"])
        for p, s in zip(inter, vf.tolist()):
            lab = s1[p][1] if s1[p][1] else s2[p][1]
            w.writerow([f"{s:.6f}", lab, p])
    print(f"[OK] 融合完成：{args.out}  共 {len(inter)} 条")

if __name__ == "__main__":
    main()
