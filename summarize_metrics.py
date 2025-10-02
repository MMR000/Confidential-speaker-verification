# summarize_metrics.py
# -*- coding: utf-8 -*-
"""
递归扫描 --root 下所有 metrics*.json（例如 metrics.json / metrics_cosine.json / metrics_snr20.0.json 等），
汇总为一张表，并导出 CSV + Markdown（可选导出 LaTeX）。

示例：
  python summarize_metrics.py \
    --root /home/mmr/MamyrModel/VoiceprintRecognition-Pytorch-develop/output/report_campp \
    --pattern "metrics*.json" \
    --out_csv summary_metrics.csv \
    --out_md  summary_metrics.md \
    --out_tex summary_metrics.tex
"""

import os
import re
import glob
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

# pandas 可选（若没有也能用内置写 CSV/Markdown）
try:
    import pandas as pd
    HAS_PD = True
except Exception:
    HAS_PD = False


def round_or_none(x: Optional[float], n=5) -> Optional[float]:
    try:
        return round(float(x), n)
    except Exception:
        return None


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def infer_group_and_variant(root: str, filepath: str) -> Tuple[str, str]:
    """
    group：root 之下的第一级子目录名（如 salt_v1 / salt_v2 / main / report_campp 本身）
    variant：从文件名推断（metrics.json -> main；metrics_cosine.json -> cosine；metrics_snr20.0.json -> snr20.0）
    """
    rel = os.path.relpath(filepath, root)
    parts = rel.split(os.sep)

    # group：取 rel 的第一段（如果 rel 形如 main/metrics.json）
    if len(parts) >= 2:
        group = parts[0]
    else:
        group = "."

    # variant：基于文件名
    fname = os.path.basename(filepath)
    m = re.match(r"metrics(?:_)?(.*)\.json$", fname)
    if m:
        v = m.group(1)
        variant = v if v else "main"
    else:
        variant = os.path.splitext(fname)[0]
    return group, variant


def collect_one(filepath: str, root: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    group, variant = infer_group_and_variant(root, filepath)

    # 基本字段
    kept     = data.get("kept")
    scores_mean = data.get("scores_mean")
    scores_std  = data.get("scores_std")
    eer     = data.get("eer")
    min_dcf = data.get("min_dcf")
    tau_eer = data.get("tau_eer")
    auc     = data.get("auc")

    # @EER 指标
    acc_eer = safe_get(data, ["metrics_at_eer", "ACC"])
    far_eer = safe_get(data, ["metrics_at_eer", "FAR"])
    frr_eer = safe_get(data, ["metrics_at_eer", "FRR"])
    f1_eer  = safe_get(data, ["metrics_at_eer", "F1"])

    # @FAR=1% / 0.1%
    acc_far1   = safe_get(data, ["metrics_at_far1", "ACC"])
    far_far1   = safe_get(data, ["metrics_at_far1", "FAR"])
    frr_far1   = safe_get(data, ["metrics_at_far1", "FRR"])

    acc_far0_1 = safe_get(data, ["metrics_at_far0_1", "ACC"])
    far_far0_1 = safe_get(data, ["metrics_at_far0_1", "FAR"])
    frr_far0_1 = safe_get(data, ["metrics_at_far0_1", "FRR"])

    # 置信区间
    eer_ci  = safe_get(data, ["bootstrap", "EER_ci95"])
    auc_ci  = safe_get(data, ["bootstrap", "AUC_ci95"])

    row = dict(
        group=group,
        variant=variant,
        file=os.path.relpath(filepath, root),
        kept=kept,
        scores_mean=round_or_none(scores_mean, 6),
        scores_std=round_or_none(scores_std, 6),
        EER=round_or_none(eer, 5),
        minDCF=round_or_none(min_dcf, 5),
        tau_EER=round_or_none(tau_eer, 5),
        AUC=round_or_none(auc, 5),
        ACC_at_EER=round_or_none(acc_eer, 4),
        FAR_at_EER=round_or_none(far_eer, 4),
        FRR_at_EER=round_or_none(frr_eer, 4),
        F1_at_EER=round_or_none(f1_eer, 4),
        ACC_at_FAR1=round_or_none(acc_far1, 4),
        FAR_at_FAR1=round_or_none(far_far1, 4),
        FRR_at_FAR1=round_or_none(frr_far1, 4),
        ACC_at_FAR0_1=round_or_none(acc_far0_1, 4),
        FAR_at_FAR0_1=round_or_none(far_far0_1, 4),
        FRR_at_FAR0_1=round_or_none(frr_far0_1, 4),
        EER_CI95=f"[{round_or_none(eer_ci[0],5)}, {round_or_none(eer_ci[1],5)}]" if isinstance(eer_ci, list) and len(eer_ci)==2 else None,
        AUC_CI95=f"[{round_or_none(auc_ci[0],5)}, {round_or_none(auc_ci[1],5)}]" if isinstance(auc_ci, list) and len(auc_ci)==2 else None
    )
    return row


def write_csv(rows: List[Dict[str, Any]], out_csv_path: str):
    cols_order = [
        "group","variant","file","kept","EER","minDCF","tau_EER","AUC",
        "ACC_at_EER","FAR_at_EER","FRR_at_EER","F1_at_EER",
        "ACC_at_FAR1","FAR_at_FAR1","FRR_at_FAR1",
        "ACC_at_FAR0_1","FAR_at_FAR0_1","FRR_at_FAR0_1",
        "scores_mean","scores_std","EER_CI95","AUC_CI95",
    ]
    if HAS_PD:
        import pandas as pd
        df = pd.DataFrame(rows)
        # 只保留我们关心的列（存在就保留）
        cols = [c for c in cols_order if c in df.columns]
        df = df[cols].sort_values(by=["group","variant","file"], ascending=True)
        df.to_csv(out_csv_path, index=False, encoding="utf-8")
    else:
        import csv
        # 收集所有列（按 cols_order 再补其余）
        all_cols = cols_order[:]
        for r in rows:
            for k in r.keys():
                if k not in all_cols:
                    all_cols.append(k)
        rows_sorted = sorted(rows, key=lambda r: (str(r.get("group")), str(r.get("variant")), str(r.get("file"))))
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_cols)
            w.writeheader()
            for r in rows_sorted:
                w.writerow(r)


def write_markdown(rows: List[Dict[str, Any]], out_md_path: str):
    # 选取论文常用列
    cols = [
        "group","variant","EER","minDCF","tau_EER","AUC",
        "ACC_at_EER","FAR_at_EER","FRR_at_EER",
        "ACC_at_FAR1","ACC_at_FAR0_1",
        "EER_CI95","AUC_CI95",
        "kept","file",
    ]
    # 按 group/variant 排序
    rows_sorted = sorted(rows, key=lambda r: (str(r.get("group")), str(r.get("variant")), str(r.get("file"))))
    # 生成 Markdown 表
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"]*len(cols)) + "|")
    for r in rows_sorted:
        vals = []
        for c in cols:
            v = r.get(c)
            if v is None:
                vals.append("")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    os.makedirs(os.path.dirname(out_md_path), exist_ok=True)
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_latex(rows: List[Dict[str, Any]], out_tex_path: str):
    # 精简列用于论文 LaTeX 表
    cols = ["group","variant","EER","minDCF","AUC","ACC_at_EER","kept"]
    header = " & ".join(cols) + " \\\\ \\hline"
    lines = [
        "\\begin{tabular}{l l r r r r r}",
        "\\hline",
        header
    ]
    rows_sorted = sorted(rows, key=lambda r: (str(r.get("group")), str(r.get("variant"))))
    for r in rows_sorted:
        vals = []
        for c in cols:
            v = r.get(c)
            if v is None:
                vals.append("")
            else:
                # 将 '_' 转义，避免 LaTeX 下划线报错
                if isinstance(v, str):
                    v = v.replace("_", "\\_")
                vals.append(str(v))
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\hline\n\\end{tabular}")

    os.makedirs(os.path.dirname(out_tex_path), exist_ok=True)
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="包含 metrics*.json 的顶层目录（递归扫描）")
    ap.add_argument("--pattern", default="metrics*.json", help="文件通配符（默认 metrics*.json）")
    ap.add_argument("--out_csv", default="summary_metrics.csv", help="导出的 CSV 文件名（相对 --root）")
    ap.add_argument("--out_md",  default="summary_metrics.md", help="导出的 Markdown 文件名（相对 --root）")
    ap.add_argument("--out_tex", default=None, help="可选：导出的 LaTeX 文件名（相对 --root）")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    files = glob.glob(os.path.join(root, "**", args.pattern), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise SystemExit(f"未在 {root} 下找到 '{args.pattern}' 文件。")

    rows: List[Dict[str, Any]] = []
    for fp in sorted(files):
        try:
            row = collect_one(fp, root)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] 解析失败：{fp} -> {e}")

    # 输出 CSV
    out_csv = os.path.join(root, args.out_csv)
    write_csv(rows, out_csv)
    print(f"[OK] CSV 导出：{out_csv}（共 {len(rows)} 行）")

    # 输出 Markdown
    out_md = os.path.join(root, args.out_md)
    write_markdown(rows, out_md)
    print(f"[OK] Markdown 导出：{out_md}")

    # 可选 LaTeX
    if args.out_tex:
        out_tex = os.path.join(root, args.out_tex)
        write_latex(rows, out_tex)
        print(f"[OK] LaTeX 导出：{out_tex}")

if __name__ == "__main__":
    main()
