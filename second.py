# -*- coding: utf-8 -*-
"""
second.py — 论文补充分析脚本（**不重跑模型与嵌入**）

用途：
  • 从现有评测产物中（scores_*.txt, metrics*.json, 曲线 csv/png）
    汇总 + 复核 + 追加论文级补充实验与可视化。
  • 绝不重新提取嵌入或做前向；仅在已有分数/标签上做后处理。

输入产物假定来自 ppsv_mpc_allinone.py（或等价流程）：
  report_dir/
    metrics.json                        # main 浮点基线
    metrics_*.json                      # 各变体（lda/euclid/plda/lenX/snrX/scaleX/...）
    scores.txt 或 scores_*.txt           # 每行：score, label(可空), path1, path2
    roc.csv / det.csv / pr.png ...     # （可选）

本脚本提供：
  1) Auto-discovery：自动发现 report_dir 下的所有 scores_*.txt 与 metrics*.json
  2) 指标复核：从分数重算 EER/minDCF/AUC/阈值，并与 metrics*.json 对照（一致性 sanity check）
  3) 配准对齐：跨变体按 (path1,path2) 对齐，用于配对自助法比较差异
  4) 置信区间与显著性：配对 bootstrap（1k 次，默认）估计 变体-基线 的 AUC/EER 差异 CI 与 p-value
  5) 风险-覆盖率曲线（Selective Verification）：以 |score-τ| 为不确定度，考察拒识带来的精度 vs 覆盖率权衡
  6) 校准补充：若可用，拟合等渗回归（isotonic）做可靠性图（Reliability）与 ECE/Brier
  7) 盐轮换补充：若同时存在 salt_v1 / salt_v2 的分数文件，做同一对 trial 的 genuine 分数相关性与 τ 稳定性
  8) 表格生成：生成论文可直接引用的 CSV/TeX（summary_no_group, full_metrics）

缺依赖将自动跳过并打印 "[SKIP]"。

示例：见文件末尾 __main__ 的 argparse 说明。
"""

import os, sys, json, math, argparse, glob, csv, itertools, time, random
from typing import List, Tuple, Dict, Optional

import numpy as np

# ---------- 可选依赖 ----------
HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    pass

HAS_SK_ISO = False
try:
    from sklearn.isotonic import IsotonicRegression
    HAS_SK_ISO = True
except Exception:
    HAS_SK_ISO = False

# ---------- 公共度量 ----------

def compute_eer_mindcf(scores: np.ndarray, labels: np.ndarray,
                       p_target: float = 0.01, c_miss: float = 1.0, c_fa: float = 1.0):
    assert scores.shape[0] == labels.shape[0]
    thr = np.unique(scores)
    thr = np.concatenate(([thr.min() - 1e-6], thr, [thr.max() + 1e-6]))
    P = int(np.sum(labels == 1)); N = int(np.sum(labels == 0))
    if P == 0 or N == 0:
        raise ValueError("全正或全负标签，无法计算 EER/minDCF")

    FPR_list, FNR_list = [], []
    for t in thr:
        pred = (scores >= t)
        TP = int(np.sum((pred == 1) & (labels == 1)))
        FP = int(np.sum((pred == 1) & (labels == 0)))
        FN = int(np.sum((pred == 0) & (labels == 1)))
        FPR_list.append(FP / float(N))
        FNR_list.append(FN / float(P))

    FPR_arr = np.asarray(FPR_list); FNR_arr = np.asarray(FNR_list)
    diff = FPR_arr - FNR_arr
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(idx) == 0:
        k = int(np.argmin(np.abs(diff))); eer = 0.5 * (FPR_arr[k] + FNR_arr[k]); tau = float(thr[k])
    else:
        k = int(idx[0]); x0, x1 = float(thr[k]), float(thr[k + 1]); y0, y1 = float(diff[k]), float(diff[k + 1])
        tau = x0 if abs(y1 - y0) < 1e-12 else x0 - y0 * (x1 - x0) / (y1 - y0)
        eer = 0.5 * (FPR_arr[k] + FNR_arr[k])

    dcf = c_miss * p_target * FNR_arr + c_fa * (1 - p_target) * FPR_arr
    min_dcf = float(np.min(dcf))
    return float(eer), float(min_dcf), tau, FPR_arr, FNR_arr, thr


def roc_points(scores: np.ndarray, labels: np.ndarray, num=2048):
    thr = np.linspace(scores.min() - 1e-6, scores.max() + 1e-6, num)
    P = np.sum(labels == 1); N = np.sum(labels == 0)
    TPR, FPR = [], []
    for t in thr:
        pred = (scores >= t)
        TP = np.sum((pred == 1) & (labels == 1))
        FP = np.sum((pred == 1) & (labels == 0))
        FN = np.sum((pred == 0) & (labels == 1))
        TPR.append(TP / float(P) if P else 0.0)
        FPR.append(FP / float(N) if N else 0.0)
    return np.asarray(FPR), np.asarray(TPR), thr


def auc_trapz(x: np.ndarray, y: np.ndarray):
    idx = np.argsort(x)
    return float(np.trapz(y[idx], x[idx]))


def find_threshold_for_far(scores: np.ndarray, labels: np.ndarray, far_target: float):
    s_unique = np.unique(scores)
    N = np.sum(labels == 0)
    best_tau = s_unique.min() - 1e-6; best_far = 1.0
    for t in s_unique:
        pred = (scores >= t)
        FP = np.sum((pred == 1) & (labels == 0))
        far = FP / float(N) if N else 0.0
        if far <= far_target:
            if (t > best_tau) or (abs(t - best_tau) < 1e-12 and far < best_far):
                best_tau = float(t); best_far = float(far)
    return best_tau


def metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, tau: float):
    pred = (scores >= tau)
    P = np.sum(labels == 1); N = np.sum(labels == 0)
    TP = int(np.sum((pred == 1) & (labels == 1)))
    FP = int(np.sum((pred == 1) & (labels == 0)))
    TN = int(np.sum((pred == 0) & (labels == 0)))
    FN = int(np.sum((pred == 0) & (labels == 1)))
    FAR = FP / float(N) if N else 0.0
    FRR = FN / float(P) if P else 0.0
    ACC = (TP + TN) / float(P + N) if (P + N) else 0.0
    PREC = TP / float(TP + FP) if (TP + FP) else 0.0
    REC = TP / float(TP + FN) if (TP + FN) else 0.0
    F1 = 2 * PREC * REC / float(PREC + REC) if (PREC + REC) else 0.0
    mu_pos = scores[labels == 1].mean() if P else 0.0
    mu_neg = scores[labels == 0].mean() if N else 0.0
    var = 0.5 * ((scores[labels == 1].var() if P else 0.0) + (scores[labels == 0].var() if N else 0.0))
    dprime = (mu_pos - mu_neg) / math.sqrt(var + 1e-12) if var > 0 else 0.0
    return dict(TP=TP, FP=FP, TN=TN, FN=FN, FAR=FAR, FRR=FRR, ACC=ACC, F1=F1, dprime=float(dprime))


def bootstrap_ci_pair(scores: np.ndarray, labels: np.ndarray, B=1000, seed=2024):
    rng = np.random.default_rng(seed)
    n = len(scores)
    eers, aucs = [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        s = scores[idx]; y = labels[idx]
        try:
            eer, _, _, _, _, _ = compute_eer_mindcf(s, y)
            FPR, TPR, _ = roc_points(s, y)
            aucs.append(auc_trapz(FPR, TPR))
            eers.append(eer)
        except Exception:
            pass
    def pct(a, p): return float(np.percentile(a, p)) if a else None
    return dict(
        EER_mean=float(np.mean(eers)) if eers else None,
        EER_ci95=[pct(eers, 2.5), pct(eers, 97.5)],
        AUC_mean=float(np.mean(aucs)) if aucs else None,
        AUC_ci95=[pct(aucs, 2.5), pct(aucs, 97.5)]
    )


def paired_bootstrap_diff(s1: np.ndarray, y: np.ndarray, s2: np.ndarray, B=1000, seed=2024):
    """配对 bootstrap：同一 trials 索引重采样，比较两种打分的差异（AUC/EER）。"""
    assert len(s1) == len(s2) == len(y)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs_auc, diffs_eer = [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        a1, a2, yy = s1[idx], s2[idx], y[idx]
        try:
            FPR1, TPR1, _ = roc_points(a1, yy)
            FPR2, TPR2, _ = roc_points(a2, yy)
            auc1, auc2 = auc_trapz(FPR1, TPR1), auc_trapz(FPR2, TPR2)
            eer1,_,_,_,_,_ = compute_eer_mindcf(a1, yy)
            eer2,_,_,_,_,_ = compute_eer_mindcf(a2, yy)
            diffs_auc.append(auc2 - auc1)
            diffs_eer.append(eer2 - eer1)
        except Exception:
            pass
    def stat(a):
        if not a: return dict(mean=None, ci95=[None,None], p=None)
        a = np.array(a)
        ci = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        mean = float(a.mean())
        # 近似双尾 p 值（原点两侧比例*2，裁剪到[0,1]）
        p = 2*min(np.mean(a>=0), np.mean(a<=0))
        p = float(min(max(p, 0.0), 1.0))
        return dict(mean=mean, ci95=ci, p=p)
    return dict(diff_auc=stat(diffs_auc), diff_eer=stat(diffs_eer))

# ---------- 读写工具 ----------

def read_scores_file(path: str):
    """返回 (scores, labels[或None], pairs[或None])。允许首行以 '#' 开头的注释头。"""
    scores, labels, pairs = [], [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'): continue
            parts = [p for p in line.replace(',', ' ').replace('\t', ' ').split() if p]
            if len(parts) < 1: continue
            try:
                s = float(parts[0])
            except Exception:
                continue
            lbl = None
            p1 = p2 = None
            if len(parts) >= 2:
                try:
                    lbl = int(float(parts[1]))
                except Exception:
                    lbl = None
            if len(parts) >= 4:
                p1, p2 = parts[2], parts[3]
            scores.append(s)
            if lbl is not None:
                labels.append(lbl)
            if p1 is not None and p2 is not None:
                pairs.append((p1, p2))
    scores = np.array(scores, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int32) if len(labels) == len(scores) else None
    pairs = pairs if len(pairs) == len(scores) else None
    return scores, labels_arr, pairs


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {path}")


def save_csv(path: str, rows: List[List]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        cw = csv.writer(f)
        for r in rows:
            cw.writerow(r)
    print(f"[SAVE] {path}")

# ---------- 可视化与补充实验 ----------

def reliability_and_ece(scores: np.ndarray, labels: np.ndarray, out_png: str, bins: int = 15):
    """若可用等渗回归，给出可靠性图与 ECE/Brier；否则用分箱频率估计。返回字典。"""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    # 将分数转为 "置信度" 的单调映射。优先：isotonic on P(y=1|s)
    if HAS_SK_ISO:
        try:
            # 原始分数可能不是 [0,1]，我们先做秩-归一化便于拟合
            order = np.argsort(scores)
            ranks = np.empty_like(order, dtype=np.float32)
            ranks[order] = np.linspace(0.0, 1.0, len(scores), endpoint=True)
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
            prob = iso.fit_transform(ranks, labels.astype(np.float32))
        except Exception as e:
            print(f"[SKIP] Isotonic 失败：{e}")
            prob = None
    else:
        prob = None

    if prob is None:
        # 后备：基于两类正态近似的似然比 -> 概率
        pos = scores[labels==1]; neg = scores[labels==0]
        mu_p, mu_n = float(np.mean(pos)), float(np.mean(neg))
        var_p, var_n = float(np.var(pos)+1e-9), float(np.var(neg)+1e-9)
        # 简化的 LLR 近似（同方差假设下等价于线性），再过 sigmoid
        w = (mu_p - mu_n) / max(var_p+var_n, 1e-9)
        b = -0.5 * (mu_p**2 - mu_n**2) / max(var_p+var_n, 1e-9)
        z = w * scores + b
        prob = 1.0 / (1.0 + np.exp(-z))

    # ECE/Brier
    bins_edges = np.linspace(0.0, 1.0, bins+1)
    ece = 0.0
    brier = float(np.mean((prob - labels)**2))
    xs, ys, ws = [], [], []
    for i in range(bins):
        m, M = bins_edges[i], bins_edges[i+1]
        idx = np.where((prob >= m) & (prob < M))[0]
        if len(idx) == 0: continue
        conf = float(np.mean(prob[idx]))
        acc  = float(np.mean(labels[idx]))
        wbin = len(idx) / float(len(labels))
        ece += abs(acc - conf) * wbin
        xs.append(conf); ys.append(acc); ws.append(wbin)
    ece = float(ece)

    if HAS_MPL:
        try:
            plt.figure(figsize=(5,5))
            plt.plot([0,1],[0,1], linestyle='--')
            plt.scatter(xs, ys, s=np.array(ws)*1200 + 10, alpha=0.7)
            plt.xlabel('Confidence (calibrated)')
            plt.ylabel('Empirical accuracy')
            plt.title(f'Reliability (ECE={ece:.4f}, Brier={brier:.4f})')
            plt.grid(True); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
        except Exception:
            pass
    return dict(ECE=ece, Brier=brier)


def risk_coverage_curve(scores: np.ndarray, labels: np.ndarray, tau: float, out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    # 覆盖率：未被拒识的比例；风险：未拒识样本上的 1-ACC
    deltas = np.linspace(0.0, max(1e-3, float(scores.max()-scores.min())/2), 50)
    cov, acc = [], []
    for d in deltas:
        keep = np.where(np.abs(scores - tau) >= d)[0]
        if len(keep) == 0:
            cov.append(0.0); acc.append(1.0); continue
        s = scores[keep]; y = labels[keep]
        pred = (s >= tau).astype(np.int32)
        a = float(np.mean(pred == y))
        cov.append(len(keep) / float(len(scores)))
        acc.append(a)
    if HAS_MPL:
        try:
            plt.figure(figsize=(5,4))
            plt.plot(cov, acc)
            plt.xlabel('Coverage'); plt.ylabel('Accuracy')
            plt.title('Risk-Coverage (abstain on |s-τ|<δ)')
            plt.grid(True); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
        except Exception:
            pass
    return dict(coverage=cov, accuracy=acc)

# ---------- 发现、对齐与汇总 ----------

def discover_variants(report_dir: str):
    scores_files = sorted(glob.glob(os.path.join(report_dir, 'scores*.txt')))
    metrics_files = sorted(glob.glob(os.path.join(report_dir, 'metrics*.json')))
    # 变体名提取规则：scores_<tag>.txt -> tag；scores.txt 记作 'main'
    def tag_from_scores(p):
        base = os.path.basename(p)
        if base == 'scores.txt': return 'main'
        t = base.replace('scores', '').replace('.txt','')
        return t[1:] if t.startswith('_') else t
    def tag_from_metrics(p):
        base = os.path.basename(p)
        t = base.replace('metrics','').replace('.json','')
        return 'main' if t=='' else (t[1:] if t.startswith('_') else t)
    variants = {}
    for p in scores_files:
        variants.setdefault(tag_from_scores(p), {})['scores'] = p
    for p in metrics_files:
        variants.setdefault(tag_from_metrics(p), {})['metrics'] = p
    return variants


def align_by_pairs(v1, v2):
    """v1/v2: (scores, labels, pairs)。对齐返回 (s1, s2, y)。若 pairs 缺失，则按索引对齐并给出警告。"""
    s1, y1, p1 = v1; s2, y2, p2 = v2
    if y1 is None or y2 is None:
        raise ValueError('需要带标签的 scores 文件用于配对比较')
    if p1 is None or p2 is None:
        n = min(len(s1), len(s2), len(y1))
        print('[WARN] 缺少 path 对，按索引对齐；请确保 trials 顺序一致。')
        return s1[:n], s2[:n], y1[:n]
    # 基于 (p1,p2) 键对齐
    map2 = {p: i for i,p in enumerate(p2)}
    idx1, idx2 = [], []
    for i, pp in enumerate(p1):
        j = map2.get(pp, None)
        if j is not None:
            idx1.append(i); idx2.append(j)
    idx1 = np.array(idx1, dtype=np.int32); idx2 = np.array(idx2, dtype=np.int32)
    return s1[idx1], s2[idx2], y1[idx1]

# ---------- 表格导出 ----------

def to_tex_table(rows: List[List[str]], align: str, caption: str, label: str):
    lines = []
    lines.append('\\begin{table}[H]')
    lines.append('\\centering')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{6pt}')
    lines.append(f'\\begin{{tabular}}{{{align}}}')
    lines.append('\\toprule')
    for i, r in enumerate(rows):
        sep = ' \\ \\midrule' if i==0 else ' \\'
        line = ' & '.join(str(x) for x in r) + sep
        lines.append(line)
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{label}}}')
    lines.append('\\end{table}')
    return "\n".join(lines)

# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report_dir', type=str, required=True, help='ppsv_mpc_allinone.py 生成的目录（含 scores_*.txt, metrics*.json）')
    ap.add_argument('--out_dir', type=str, default=None, help='输出目录（默认 report_dir/paper_addons）')
    ap.add_argument('--baseline', type=str, default='cosine', help='与之比较的基线变体名（默认 cosine；也可用 main/lda/…）')
    ap.add_argument('--bootstrap', type=int, default=1000, help='bootstrap 次数（差异检验与 CI）')
    ap.add_argument('--do_all', action='store_true', help='一键运行所有补充分析与表格')
    ap.add_argument('--variants', nargs='*', default=None, help='仅分析这些变体（默认自动发现全部）')
    args = ap.parse_args()

    report_dir = args.report_dir
    out_dir = args.out_dir or os.path.join(report_dir, 'paper_addons')
    os.makedirs(out_dir, exist_ok=True)

    variants = discover_variants(report_dir)
    if not variants:
        print('[ERROR] 未发现 scores_*.txt / metrics*.json'); sys.exit(1)

    # 子集选择
    if args.variants:
        variants = {k:v for k,v in variants.items() if k in set(args.variants)}
        if not variants:
            print('[ERROR] 所选变体皆不存在。'); sys.exit(1)

    print(f"[INFO] 发现变体：{sorted(variants.keys())}")

    # 读取所有 scores + 计算/复核指标
    cache = {}
    summary_rows = [['Variant','EER','minDCF','AUC','ACC@EER','Trials']]

    for tag, files in variants.items():
        scores_path = files.get('scores', None)
        metrics_path = files.get('metrics', None)
        if not scores_path:
            print(f"[SKIP] {tag}: 缺 scores_*.txt，后续比较将不可用。")
            continue
        s, y, pairs = read_scores_file(scores_path)
        if y is None:
            print(f"[SKIP] {tag}: 分数文件缺标签，跳过指标复核与后续分析。")
            continue
        # 计算指标
        eer, mindcf, tau, _, _, _ = compute_eer_mindcf(s, y)
        FPR, TPR, _ = roc_points(s, y); auc = auc_trapz(FPR, TPR)
        at_eer = metrics_at_threshold(s, y, tau)
        summary_rows.append([tag, f"{eer:.5f}", f"{mindcf:.5f}", f"{auc:.5f}", f"{at_eer['ACC']:.4f}", str(len(s))])
        cache[tag] = dict(scores=s, labels=y, pairs=pairs, tau=tau, auc=auc, eer=eer, mindcf=mindcf)

        # 与 metrics*.json 对照（若存在）
        if metrics_path:
            try:
                M = json.load(open(metrics_path,'r',encoding='utf-8'))
                m_eer = M.get('eer', None)
                m_auc = M.get('auc', None)
                if m_eer is not None and abs(m_eer - eer) > 5e-3:
                    print(f"[WARN] {tag}: 复核 EER={eer:.4f} 与 metrics.json 中 {m_eer:.4f} 差异较大，请确认 trials 顺序或文件版本。")
            except Exception as e:
                print(f"[SKIP] 读取 {metrics_path} 失败：{e}")

        # 置信区间（pairs bootstrap）
        boot = bootstrap_ci_pair(s, y, B=args.bootstrap, seed=2024)
        cache[tag]['boot'] = boot
        save_json(os.path.join(out_dir, f'metrics_boot_{tag}.json'), boot)

        # 风险-覆盖率曲线
        rc = risk_coverage_curve(s, y, tau, os.path.join(out_dir, f'risk_coverage_{tag}.png'))
        save_json(os.path.join(out_dir, f'risk_coverage_{tag}.json'), rc)

        # 可靠性图 & ECE
        try:
            rel = reliability_and_ece(s, y, os.path.join(out_dir, f'reliability_{tag}.png'))
            save_json(os.path.join(out_dir, f'reliability_{tag}.json'), rel)
        except Exception as e:
            print(f"[SKIP] reliability {tag}: {e}")

    # 导出 summary 表
    save_csv(os.path.join(out_dir, 'summary_no_group.csv'), summary_rows)
    tex = to_tex_table(summary_rows, align='l r r r r r', caption='Summary without group column.', label='tab:summary-no-group')
    with open(os.path.join(out_dir, 'summary_no_group.tex'), 'w', encoding='utf-8') as f:
        f.write(tex)
    print(f"[SAVE] {os.path.join(out_dir, 'summary_no_group.tex')}")

    # 与基线的配对差异检验
    base = args.baseline
    if base not in cache:
        print(f"[SKIP] baseline '{base}' 不在可比较集合中（当前可用：{list(cache.keys())}）")
    else:
        for tag in sorted(cache.keys()):
            if tag == base: continue
            try:
                s1,y1,p1 = cache[base]['scores'], cache[base]['labels'], cache[base]['pairs']
                s2,y2,p2 = cache[tag]['scores'],  cache[tag]['labels'],  cache[tag]['pairs']
                a1,a2,yy = align_by_pairs((s1,y1,p1),(s2,y2,p2))
                diff = paired_bootstrap_diff(a1, yy, a2, B=args.bootstrap, seed=2024)
                save_json(os.path.join(out_dir, f'paired_diff_{tag}_vs_{base}.json'), diff)
            except Exception as e:
                print(f"[SKIP] 配对差异 {tag} vs {base}: {e}")

    # 盐轮换补充（若存在）
    salt_tags = [t for t in cache.keys() if 'salt' in t]
    if len(salt_tags) >= 2:
        # 取前两把盐
        t1, t2 = salt_tags[0], salt_tags[1]
        print(f"[INFO] 盐轮换补充：比较 {t1} 与 {t2}")
        try:
            s1,y1,p1 = cache[t1]['scores'], cache[t1]['labels'], cache[t1]['pairs']
            s2,y2,p2 = cache[t2]['scores'], cache[t2]['labels'], cache[t2]['pairs']
            a1,a2,yy = align_by_pairs((s1,y1,p1),(s2,y2,p2))
            # 只看 genuine 的相关性
            idx = np.where(yy==1)[0]
            if len(idx) > 3:
                r = float(np.corrcoef(a1[idx], a2[idx])[0,1])
            else:
                r = float('nan')
            out = dict(salt_a=t1, salt_b=t2, tau_a=cache[t1]['tau'], tau_b=cache[t2]['tau'], corr_genuine=r)
            save_json(os.path.join(out_dir, f'salt_compare_{t1}_vs_{t2}.json'), out)
        except Exception as e:
            print(f"[SKIP] 盐轮换补充失败：{e}")
    else:
        print('[INFO] 未检测到 >=2 个 salt_* 变体，跳过盐轮换补充。')

    print('[DONE] 补充分析完成。产物位于：', out_dir)


if __name__ == '__main__':
    main()
