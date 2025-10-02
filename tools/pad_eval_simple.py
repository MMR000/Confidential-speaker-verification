# tools/pad_eval_simple.py
import argparse, os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

def _safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    # 单类保护
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)

def _eer(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    fpr, tpr, th = roc_curve(y_true, y_score)  # 正类=1
    fnr = 1 - tpr
    # 找 FPR 与 FNR 最接近点
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    tau = th[idx]
    return float(eer), float(tau)

def _subset_mask(paths, contains):
    c = contains.lower()
    return np.array([c in str(p).lower() for p in paths], dtype=bool)

def plot_curves(y_true, y_score, out_prefix):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.grid(True)
    plt.savefig(out_prefix + "_roc.png", dpi=160, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR")
    plt.grid(True)
    plt.savefig(out_prefix + "_pr.png", dpi=160, bbox_inches="tight"); plt.close()

def eval_one(y_true, y_score, tag, outdir):
    os.makedirs(outdir, exist_ok=True)
    auc = _safe_auc(y_true, y_score)
    if auc is None:
        print(f"[SKIP] {tag}: 只有单类样本，跳过 AUC/EER 计算。")
        return None
    eer, tau = _eer(y_true, y_score)
    print(f"[{tag}] AUC={auc:.5f}  EER={eer:.5f}  tau(EER)={tau:.5f}")
    plot_curves(y_true, y_score, os.path.join(outdir, tag))
    # 保存指标
    with open(os.path.join(outdir, f"{tag}_metrics.txt"), "w") as f:
        f.write(f"AUC\t{auc:.6f}\nEER\t{eer:.6f}\nTAU_EER\t{tau:.6f}\n")
    return {"AUC": auc, "EER": eer, "TAU_EER": tau}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True)
    ap.add_argument("--scores", required=True)  # 需含 'score' 与 'path'
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    trials = pd.read_csv(args.trials, sep="\t")
    scores = pd.read_csv(args.scores, sep="\t")

    # 对齐（以 path 为键）
    if "path" not in scores.columns:
        # 兼容无表头 4 列：score, label(opt), path1, path2（取 path1）
        if scores.shape[1] >= 3:
            scores.columns = ["score", "label(optional)", "path", *[f"x{i}" for i in range(scores.shape[1]-3)]]
        else:
            raise RuntimeError("scores 必须包含 path 列或至少 3 列（含路径）。")

    # trials 拿 path/label
    if "path" not in trials.columns:
        path_col = trials.columns[1]
    else:
        path_col = "path"
    if "label" not in trials.columns:
        label_col = trials.columns[0]
    else:
        label_col = "label"

    df = pd.merge(trials[[path_col, label_col]], scores[["path", "score"]], left_on=path_col, right_on="path", how="inner")
    # label 统一成 1(bonafide), 0(spoof)
    if df[label_col].dtype == object:
        df["y_true"] = df[label_col].astype(str).str.lower().map(lambda s: 1 if "real" in s or "bona" in s else 0)
    else:
        df["y_true"] = df[label_col].astype(int)
    df["y_score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["y_score"])
    df = df.reset_index(drop=True)

    os.makedirs(args.outdir, exist_ok=True)
    # 总体
    eval_one(df["y_true"], df["y_score"], "overall", args.outdir)

    # 可用分组：仅当两类都存在时才评
    # real vs Synthesized
    m_real = _subset_mask(df["path"], "real")
    m_syn  = _subset_mask(df["path"], "Synthesized")
    m_pair = m_real | m_syn
    if m_pair.sum() > 0:
        eval_one(df.loc[m_pair, "y_true"], df.loc[m_pair, "y_score"], "real_vs_synthesized", args.outdir)

    print(f"[DONE] 输出目录：{args.outdir}")

if __name__ == "__main__":
    main()
