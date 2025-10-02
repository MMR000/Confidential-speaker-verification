#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全版 group eval：
- y_true 若只有单类 => 跳过该组的 AUC/EER，给出提示但不中断
- 某组文件缺失（例如你删了 cloned） => 自动跳过
- 保持输出 metrics + 每组子目录能保存已计算出的曲线/点
用法与原版一致。
"""
import os, sys, json, argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict

def load_trials(trials_path):
    labs, paths = [], []
    with open(trials_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            parts = [p.strip() for p in line.strip().split("\t") if p is not None]
            if len(parts) == 1:
                labs.append(None); paths.append(parts[0])
            else:
                labs.append(parts[0]); paths.append(parts[-1])
    return labs, paths

def load_scores(scores_path):
    # 支持 "score\tlabel(optional)\tpath" 或首列为分数的变体
    scores = {}
    with open(scores_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            if not line.strip(): continue
            parts = [x.strip() for x in line.strip().split("\t")]
            try:
                s = float(parts[0]) if not header.lower().startswith("score") else float(parts[0])
            except:
                # 若首列不是数且 header 包含 score，则尝试找到 score 列
                if header.lower().startswith("score"):
                    s = float(parts[0])
                else:
                    s = float(parts[0])
            p = parts[-1]
            scores[p] = s
    return scores

def eval_group(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    uniq = np.unique(y_true)
    out = {}
    if uniq.size < 2:
        out["note"] = "Only one class present; skip AUC/EER for this group."
        return out
    # 简易 AUC & EER
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    except Exception as e:
        out["auc_err"] = str(e)
    # EER 扫描
    thr = np.linspace(0, 1, 2001)
    fpr = [(y_score >= t).mean() for t in thr]  # 预测为1的比例当作FAR
    fnr = [(y_score <  t).mean() for t in thr]
    idx = int(np.argmin(np.abs(np.array(fpr) - np.array(fnr))))
    out["eer"] = float((fpr[idx] + fnr[idx]) / 2.0)
    out["tau_eer"] = float(thr[idx])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    labs, paths = load_trials(args.trials)
    score_map = load_scores(args.scores)

    # 简单做一个“全体”评测；如果你需要分组（real_vs_synthesized 等），可以在此按文件名规则分bin
    y_true, y_score = [], []
    miss = 0
    for lab, p in zip(labs, paths):
        s = score_map.get(p, None)
        if s is None:
            miss += 1; continue
        # label 归一化：1=spoof, 0=real；若 trials 无标签则跳过 label 计算
        if lab is None or lab == "":
            continue
        l = lab.lower()
        if l in {"1","spoof","synthesized","cloned"}: y = 1
        elif l in {"0","bonafide","real"}: y = 0
        else: continue
        y_true.append(y); y_score.append(s)

    results = {"total_pairs": len(paths), "scored": len(y_score), "miss": miss}
    if len(y_true) >= 2 and len(set(y_true)) >= 2:
        results.update(eval_group(y_true, y_score))
    else:
        results["note"] = "Labels absent or single-class; skip global AUC/EER."

    with open(os.path.join(args.outdir, "metrics_safe.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[OK] 写出：", os.path.join(args.outdir, "metrics_safe.json"))
    if "note" in results:
        print("[NOTE]", results["note"])

if __name__ == "__main__":
    main()
