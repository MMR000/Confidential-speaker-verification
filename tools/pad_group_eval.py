#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAD group eval (robust)
- 自动识别 scores/trials 是否有表头与分隔符
- 支持行首注释(#)，支持有/无 header
- 自动寻找路径列与标签列；若找不到则回退到顺序对齐
- 先按路径对齐，不成再尝试顺序对齐（并明确报警）
- 计算整体 AUC/EER，以及 real_vs_cloned / real_vs_Synthesized 子集的指标
- 产出 JSON 与 ROC/PR 图
"""

import os
import re
import json
import argparse
from io import StringIO
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

# -----------------------
# 小工具
# -----------------------
def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _read_clean_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.readlines()
    # 去掉空行与注释
    lines = [ln for ln in raw if ln.strip() and not ln.lstrip().startswith("#")]
    return lines

def _guess_sep(lines, default="\t"):
    # 若包含制表符优先认为是 TSV，否则尝试逗号，否则空白
    head = "".join(lines[:10])
    if "\t" in head:
        return "\t"
    if "," in head and (head.count(",") >= head.count(" ")):
        return ","
    return r"\s+"

def _looks_like_header(cols):
    # 有字母/典型列名就当有表头
    joined = " ".join([str(c) for c in cols]).lower()
    has_alpha = any(any(ch.isalpha() for ch in str(c)) for c in cols)
    has_keywords = any(k in joined for k in ["score", "path", "wav", "utt", "file", "label", "cls", "target"])
    return has_alpha or has_keywords

def _normalize_path(p):
    if pd.isna(p):
        return None
    p = str(p).strip()
    if not p:
        return None
    # 处理一些常见尾巴：比如被追加的 ".1" / ".2" 之类的切段后缀
    # 只在它像 ".wav.1" 这样的模式时尝试去掉最后的 ".数字"
    if re.search(r"\.(wav|flac|mp3|m4a)\.\d+$", p, flags=re.IGNORECASE):
        p = re.sub(r"(\.(wav|flac|mp3|m4a))\.\d+$", r"\1", p, flags=re.IGNORECASE)
    # 标准化大小写与绝对路径（大小写不强制，因为 Linux 大小写敏感；仅 strip）
    try:
        p = os.path.abspath(p)
    except Exception:
        pass
    return p

def _to_numeric_series(s, name):
    # 把分数列安全转成 float，无法解析的置为 NaN
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        # 某些极端情况（例如列名被错误识别），再兜底一次
        return pd.Series([pd.to_numeric(x, errors="coerce") for x in s], name=name)

def _col_in(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    return None

def _infer_label_col(df):
    return _col_in(df, ["label", "y", "target", "cls"])

def _infer_path_col(df):
    # 优先 path1 -> path -> wav -> file...
    return _col_in(df, ["path1", "path", "wav", "wav_path", "utt", "utt_path", "file", "filepath"])

def _infer_score_col(df):
    # 常见命名: score / prob / logit
    return _col_in(df, ["score", "prob", "logit"])

def _map_label(x):
    # 将文本标签映射为 1/0：real/bonafide/bonafied 视为 1；cloned/synthesized/spoof 视为 0
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "bonafide", "bona", "real", "genuine"}:
        return 1
    if s in {"0", "false", "spoof", "fake", "cloned", "synthesized", "synth", "tts"}:
        return 0
    # 既不是 1 也不是 0，就尝试数字
    try:
        v = float(s)
        if v in (0.0, 1.0):
            return int(v)
    except Exception:
        pass
    return None

def _compute_eer(y_true, y_score):
    # 这里 y_true：1=bonafide, 0=spoof
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    # 找离交点最近点
    idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    tau = thr[idx]
    return float(eer), float(tau)

def _plot_roc_pr(y_true, y_score, title, out_prefix):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.5f}")
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
    plt.xlabel("FAR")
    plt.ylabel("TAR")
    plt.title(f"ROC - {title}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_prefix + "_roc.png", dpi=160)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec, lw=2, label=f"AP={ap:.5f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR - {title}")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_prefix + "_pr.png", dpi=160)
    plt.close()

# -----------------------
# 读取 trials
# -----------------------
def load_trials(trials_path, trial_path_col=None, trial_label_col=None):
    lines = _read_clean_lines(trials_path)
    if not lines:
        raise RuntimeError(f"trials 文件为空：{trials_path}")

    sep = _guess_sep(lines)
    # 尝试有表头
    df_try = pd.read_csv(StringIO("".join(lines)), sep=sep, engine="python", dtype=str, header=0)
    if _looks_like_header(df_try.columns):
        df = df_try
    else:
        df = pd.read_csv(StringIO("".join(lines)), sep=sep, engine="python", dtype=str, header=None)

    # 统一小写列名
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 如果无列名且列数很典型，赋默认名
    if set(df.columns) == set(range(len(df.columns))):
        n = df.shape[1]
        if n == 1:
            df.columns = ["path"]
        elif n == 2:
            df.columns = ["label", "path"]
        elif n >= 3:
            df.columns = ["label", "path1", "path2"] + [f"col{i}" for i in range(4, n+1)]

    # 允许显式指定列
    if trial_path_col:
        pcol = trial_path_col.lower()
        if pcol not in df.columns:
            raise RuntimeError(f"trials 中找不到指定的路径列：{trial_path_col}；现有列：{list(df.columns)}")
        path_col = pcol
    else:
        path_col = _infer_path_col(df)
        if path_col is None:
            # 若没有路径列且只有一列，就当这一列是路径
            if df.shape[1] == 1:
                path_col = df.columns[0]
            else:
                # 兜底：把第二列当路径
                path_col = df.columns[min(1, df.shape[1]-1)]

    if trial_label_col:
        lcol = trial_label_col.lower()
        if lcol not in df.columns:
            raise RuntimeError(f"trials 中找不到指定的标签列：{trial_label_col}；现有列：{list(df.columns)}")
        label_col = lcol
    else:
        label_col = _infer_label_col(df)

    # 生成归一化路径与标签
    trial_paths = df[path_col].map(_normalize_path).tolist()
    if label_col is not None:
        y_true = df[label_col].map(_map_label).tolist()
    else:
        # 无标签列：从路径猜（包含 real/bona 则为 1；cloned/tts/synth 则为 0；其他 None）
        y_true = []
        for p in trial_paths:
            s = (p or "").lower()
            if "real" in s or "bona" in s:
                y_true.append(1)
            elif "cloned" in s or "synth" in s or "tts" in s or "spoof" in s or "fake" in s:
                y_true.append(0)
            else:
                y_true.append(None)

    # 识别子集（通过路径）
    groups = []
    for p in trial_paths:
        s = (p or "").lower()
        if "cloned" in s:
            groups.append("cloned")
        elif "synth" in s or "tts" in s:
            groups.append("Synthesized")
        elif "real" in s or "bona" in s or "genuine" in s:
            groups.append("real")
        else:
            groups.append("unknown")

    return trial_paths, y_true, groups

# -----------------------
# 读取 scores 并与 trial 对齐
# -----------------------
def load_scores_aligned(scores_path, trial_paths, scores_path_col=None):
    lines = _read_clean_lines(scores_path)
    if not lines:
        raise RuntimeError(f"scores 文件为空：{scores_path}")

    sep = _guess_sep(lines)

    # 先按“有表头”读
    df_try = pd.read_csv(StringIO("".join(lines)), sep=sep, engine="python", dtype=str, header=0)
    df = None
    if _looks_like_header(df_try.columns):
        df = df_try
    else:
        # 无表头：按无表头重读并赋名
        df = pd.read_csv(StringIO("".join(lines)), sep=sep, engine="python", dtype=str, header=None)
        n = df.shape[1]
        if n == 1:
            df.columns = ["score"]
        elif n == 2:
            df.columns = ["score", "label"]
        elif n == 3:
            df.columns = ["score", "label", "path1"]
        elif n >= 4:
            names = ["score", "label", "path1", "path2"]
            names += [f"col{i}" for i in range(5, n+1)]
            df.columns = names

    df.columns = [str(c).strip().lower() for c in df.columns]

    # 寻找分数列
    score_col = _infer_score_col(df)
    if score_col is None:
        # 若第一列看起来全是数字，就把第一列当 score
        first = df.columns[0]
        if pd.to_numeric(df[first], errors="coerce").notna().mean() > 0.9:
            score_col = first
        else:
            raise RuntimeError(f"scores 文件缺少 'score/prob/logit' 列，且第一列不像分数。列名：{list(df.columns)}")

    # 寻找路径列
    if scores_path_col:
        pcol = scores_path_col.lower()
        if pcol not in df.columns:
            raise RuntimeError(f"scores 中找不到指定的路径列：{scores_path_col}；现有列：{list(df.columns)}")
        path_col = pcol
    else:
        path_col = _infer_path_col(df)

    # 清洗分数
    df[score_col] = _to_numeric_series(df[score_col], score_col)

    # 先试【按路径对齐】
    if path_col is not None:
        df["_norm_path"] = df[path_col].map(_normalize_path)
        # 有些分数文件同时有 path1/ path2：优先 path1
        if path_col != "path1" and "path1" in df.columns:
            df["_norm_path"] = df["path1"].map(_normalize_path)

        # 对于相同路径的多条记录，取均值
        grp = df.groupby("_norm_path")[score_col].mean()

        aligned = []
        miss = 0
        for p in trial_paths:
            sp = grp.get(p, np.nan)
            if pd.isna(sp):
                miss += 1
            aligned.append(sp)

        if miss <= 0.01 * len(trial_paths):
            # 可以接受的缺失（<=1%）
            return np.array(aligned, dtype=float)

        # 路径对不上，做一些宽松尝试：basename 匹配
        grp_base = df.copy()
        grp_base["_base"] = grp_base["_norm_path"].map(lambda x: os.path.basename(x) if isinstance(x, str) else None)
        gb = grp_base.groupby("_base")[score_col].mean()
        miss2 = 0
        aligned2 = []
        for p in trial_paths:
            b = os.path.basename(p) if isinstance(p, str) else None
            sp = gb.get(b, np.nan)
            if pd.isna(sp):
                miss2 += 1
            aligned2.append(sp)
        if miss2 <= 0.01 * len(trial_paths):
            return np.array(aligned2, dtype=float)

        print(f"[WARN] 按路径对齐失败：缺失 {miss}/{len(trial_paths)}；basename 仍缺失 {miss2}。将回退到顺序对齐。")

    # 回退【顺序对齐】
    if len(df) < len(trial_paths):
        raise RuntimeError(f"[顺序对齐失败] scores 行数({len(df)}) < trials 行数({len(trial_paths)})。")
    return df[score_col].astype(float).values[:len(trial_paths)]

# -----------------------
# 主流程
# -----------------------
def evaluate_and_save(y_true, y_score, title, outdir, tag):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    mask = np.isfinite(y_score) & (~pd.isna(y_true))
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask].astype(float)

    if len(y_true) == 0:
        raise RuntimeError(f"{tag}: 有效样本数为 0，无法评测。")

    auc = roc_auc_score(y_true, y_score)
    eer, tau = _compute_eer(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    _plot_roc_pr(y_true, y_score, f"{title}", os.path.join(outdir, f"{tag}"))

    return {
        "samples": int(len(y_true)),
        "AUC": float(auc),
        "EER": float(eer),
        "tau_at_EER": float(tau),
        "AP": float(ap),
    }

def main():
    parser = argparse.ArgumentParser(description="PAD group evaluation (robust)")
    parser.add_argument("--trials", required=True, help="pad_trials_xxx.tsv")
    parser.add_argument("--scores", required=True, help="scores_pad.txt（任意TSV/CSV/空白分隔，带/不带表头均可）")
    parser.add_argument("--outdir", required=True, help="输出目录")
    # 可选：手动指定列名
    parser.add_argument("--trial_path_col", default=None, help="trials 的路径列名（可选）")
    parser.add_argument("--trial_label_col", default=None, help="trials 的标签列名（可选）")
    parser.add_argument("--scores_path_col", default=None, help="scores 的路径列名（可选）")
    args = parser.parse_args()

    _ensure_dir(args.outdir)

    # 1) trials
    trial_paths, y_true_raw, groups = load_trials(
        args.trials, trial_path_col=args.trial_path_col, trial_label_col=args.trial_label_col
    )

    # 2) scores 对齐
    y_score = load_scores_aligned(args.scores, trial_paths, scores_path_col=args.scores_path_col)

    # 3) 将标签中的 None 去掉（仅用于总体/子集时通过掩码处理）
    y_true = np.array([np.nan if v is None else int(v) for v in y_true_raw], dtype=float)
    groups = np.array(groups, dtype=object)
    trial_paths = np.array(trial_paths, dtype=object)

    # 4) 汇总评测
    results = {}

    # Overall
    overall_mask = np.isfinite(y_true)
    results["overall"] = evaluate_and_save(
        y_true[overall_mask], y_score[overall_mask],
        title="Overall (1=bonafide, 0=spoof)", outdir=args.outdir, tag="overall"
    )

    # real_vs_cloned：所有 real + cloned
    mask_rc = ((groups == "real") | (groups == "cloned")) & np.isfinite(y_true)
    if mask_rc.sum() > 0:
        results["real_vs_cloned"] = evaluate_and_save(
            y_true[mask_rc], y_score[mask_rc],
            title="Real vs Cloned", outdir=args.outdir, tag="real_vs_cloned"
        )

    # real_vs_Synthesized：所有 real + Synthesized
    mask_rs = ((groups == "real") | (groups == "Synthesized")) & np.isfinite(y_true)
    if mask_rs.sum() > 0:
        results["real_vs_Synthesized"] = evaluate_and_save(
            y_true[mask_rs], y_score[mask_rs],
            title="Real vs Synthesized", outdir=args.outdir, tag="real_vs_Synthesized"
        )

    # 5) 额外信息与保存
    meta = {
        "trials": os.path.abspath(args.trials),
        "scores": os.path.abspath(args.scores),
        "counts": {
            "total": int(len(y_true)),
            "valid": int(np.isfinite(y_true).sum()),
            "by_group": {
                "real": int((groups == "real").sum()),
                "cloned": int((groups == "cloned").sum()),
                "Synthesized": int((groups == "Synthesized").sum()),
                "unknown": int((groups == "unknown").sum()),
            },
        },
    }
    with open(os.path.join(args.outdir, "metrics_group.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)

    print("[OK] 已保存分组评测：", os.path.join(args.outdir, "metrics_group.json"))
    for k, v in results.items():
        print(f"  - {k}: AUC={v['AUC']:.5f}, EER={v['EER']:.5f}, AP={v['AP']:.5f}, tau@EER={v['tau_at_EER']:.4f}")

if __name__ == "__main__":
    main()
