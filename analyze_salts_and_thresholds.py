# analyze_salts_and_thresholds.py
# -*- coding: utf-8 -*-
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_scores(path):
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) < 4:
                parts = line.split()
                if len(parts) < 4: continue
            try:
                s = float(parts[0])
            except:
                continue
            lbl = parts[1]
            lbl = int(lbl) if lbl not in ("", None) else -1
            a, b = parts[2], parts[3]
            k1, k2 = sorted([a, b])
            key = k1 + "||" + k2
            data.append((key, s, lbl))
    return data

def align(scores1, scores2):
    d1 = {k:(s,l) for k,s,l in scores1}
    d2 = {k:(s,l) for k,s,l in scores2}
    common = sorted(set(d1.keys()) & set(d2.keys()))
    s1 = np.array([d1[k][0] for k in common], dtype=np.float64)
    s2 = np.array([d2[k][0] for k in common], dtype=np.float64)
    y  = np.array([d1[k][1] if d1[k][1]!=-1 else d2[k][1] for k in common], dtype=np.int32)
    return s1, s2, y, common

def pearsonr(x, y):
    x = (x - x.mean()); y = (y - y.mean())
    denom = (np.linalg.norm(x)*np.linalg.norm(y) + 1e-12)
    return float(np.dot(x, y)/denom)

def _rank(a):
    idx = np.argsort(a)
    r = np.empty_like(idx, dtype=np.float64)
    r[idx] = np.arange(len(a))
    return r

def spearmanr(x, y):
    rx, ry = _rank(x), _rank(y)
    return pearsonr(rx, ry)

def kendall_tau(x, y):
    n = len(x)
    if n > 6000:
        n = 6000
        idx = np.random.RandomState(0).choice(len(x), n, replace=False)
        x, y = x[idx], y[idx]
    s = 0.0; tot = 0.0
    for i in range(n):
        d = np.sign(x[i] - x[i+1:]) * np.sign(y[i] - y[i+1:])
        s += d.sum(); tot += len(d)
    return float(s/(tot+1e-12))

def ks_statistic(x, y):
    xs = np.sort(x); ys = np.sort(y)
    allv = np.sort(np.unique(np.concatenate([xs, ys])))
    ix = 0; iy = 0; nx = len(xs); ny = len(ys)
    dmax = 0.0
    for v in allv:
        while ix < nx and xs[ix] <= v: ix += 1
        while iy < ny and ys[iy] <= v: iy += 1
        fx = ix/float(nx); fy = iy/float(ny)
        dmax = max(dmax, abs(fx - fy))
    return float(dmax)

def js_divergence(x, y, bins=100):
    lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
    if lo == hi:
        return 0.0
    px, _ = np.histogram(x, bins=bins, range=(lo,hi), density=True)
    py, _ = np.histogram(y, bins=bins, range=(lo,hi), density=True)
    px = px/px.sum(); py = py/py.sum()
    m = 0.5*(px + py)
    def kl(p, q):
        p = np.clip(p, 1e-12, 1.0); q = np.clip(q, 1e-12, 1.0)
        return float(np.sum(p*np.log(p/q)))
    return 0.5*kl(px, m) + 0.5*kl(py, m)

def plot_scatter_hist(s1, s2, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(5,5))
    plt.scatter(s1, s2, s=5, alpha=0.5)
    plt.xlabel("score (salt v1)"); plt.ylabel("score (salt v2)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "salt_scatter.png"), dpi=200); plt.close()

    plt.figure(figsize=(6,4))
    bins = 60
    lo = float(min(s1.min(), s2.min())); hi = float(max(s1.max(), s2.max()))
    plt.hist(s1, bins=bins, range=(lo,hi), alpha=0.6, density=True, label="salt v1")
    plt.hist(s2, bins=bins, range=(lo,hi), alpha=0.6, density=True, label="salt v2")
    plt.xlabel("score"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "salt_hist_overlay.png"), dpi=200); plt.close()

    plt.figure(figsize=(6,4))
    diff = s2 - s1
    plt.hist(diff, bins=60, alpha=0.8, density=True)
    plt.xlabel("score_diff (v2 - v1)"); plt.ylabel("density"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "salt_diff_hist.png"), dpi=200); plt.close()

def find_threshold_for_far(scores, labels, far_target):
    s_unique = np.unique(scores)
    N = np.sum(labels==0)
    best_tau = s_unique.min()-1e-6; best_far = 1.0
    for t in s_unique:
        pred = (scores >= t)
        FP = np.sum((pred==1) & (labels==0))
        far = FP/float(N) if N else 0.0
        if far <= far_target and far <= best_far:
            best_far = far; best_tau = float(t)
    return best_tau

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores1", required=True)
    ap.add_argument("--scores2", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--outdir",  required=True)
    args = ap.parse_args()

    s1_raw = load_scores(args.scores1)
    s2_raw = load_scores(args.scores2)
    s1, s2, y, _ = align(s1_raw, s2_raw)
    if len(s1)==0:
        raise SystemExit("两份 scores 无对齐样本：确认两次都用了同一份 auto_trials.txt。")

    p = pearsonr(s1, s2)
    sp = spearmanr(s1, s2)
    kt = kendall_tau(s1, s2)
    ks = ks_statistic(s1, s2)
    js = js_divergence(s1, s2)

    os.makedirs(args.outdir, exist_ok=True)
    plot_scatter_hist(s1, s2, args.outdir)

    print("[SALT-COMPARE] aligned pairs =", len(s1))
    print(f"[SALT-COMPARE] Pearson r  = {p:.4f}")
    print(f"[SALT-COMPARE] Spearman   = {sp:.4f}")
    print(f"[SALT-COMPARE] Kendall τ  = {kt:.4f}")
    print(f"[SALT-COMPARE] KS stat    = {ks:.4f}")
    print(f"[SALT-COMPARE] JS div     = {js:.4f}")
    if abs(p) < 0.2 and ks > 0.2:
        print("[SALT-COMPARE] 结论：跨盐低相关且分布差异明显，支持不可链接/可撤销。")

    with open(args.metrics, "r", encoding="utf-8") as f:
        m = json.load(f)
    tau_eer = float(m.get("tau_eer", 0.5))

    tau_far1 = None; tau_far0_1 = None
    if np.all((y==0) | (y==1)):
        tau_far1   = find_threshold_for_far(s1, y, 0.01)
        tau_far0_1 = find_threshold_for_far(s1, y, 0.001)

    out = {"tau_eer": float(tau_eer),
           "tau_far1": float(tau_far1) if tau_far1 is not None else None,
           "tau_far0_1": float(tau_far0_1) if tau_far0_1 is not None else None}
    with open(os.path.join(args.outdir, "deploy_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[CALIB] 阈值导出 ->", os.path.join(args.outdir, "deploy_thresholds.json"))

if __name__ == "__main__":
    main()
