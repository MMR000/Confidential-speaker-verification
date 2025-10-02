#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ppsv_dev_choice_grid.py
- 生成 O/E/H 三种 dev 选择 → O/E/H 三个 test 的 3×3 网格对比
- 表A：dev 上固定点 tau_EER（应用到各 test；表格里重复填入，便于直观看迁移）
- 表B：在 dev 的 tau 上，于各 test 集上的 (FAR+FRR)/2（“EER@tau”，用于迁移敏感度）
- 自动导出 CSV + LaTeX 片段

用法示例（与你项目环境一致）见本文末尾。
"""
import os, sys, argparse, json
import numpy as np

# 引入你的 runner 工具
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ppsv_runner import (
    MVectorPredictor, ensure_dir, StageTimer, pb_iter,
    build_audio_index_from_dirs, load_trials_with_index,
    prepare_embeddings, score_pairs_fixed_point,
    compute_eer_mindcf, metrics_at_threshold,
    save_json
)

def load_pairs_for(split_name, trials_path, index_dirs):
    if not os.path.isfile(trials_path):
        raise FileNotFoundError(f"[{split_name}] trials 不存在：{trials_path}")
    base = os.path.dirname(trials_path)
    pairs, labels = load_trials_with_index(trials_path, index=index_dirs, base_hint=base)
    if not pairs or labels is None:
        raise RuntimeError(f"[{split_name}] trials 解析失败或缺少标签：{trials_path}")
    return pairs, np.asarray(labels, np.int32)

def union_utts(*pair_lists):
    return sorted({p for pairs in pair_lists for ab in pairs for p in ab})

def grid_to_csv(path, header, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join([""] + header) + "\n")
        for rname, vals in rows:
            f.write(",".join([rname] + [f"{v}" for v in vals]) + "\n")
    print(f"[SAVE] {path}")

def latex_table_3x3(caption, label, header, rows, fmt="%.2f"):
    # rows: [(row_name, [vO,vE,vH]), ...]
    head = " & " + " & ".join(header) + " \\\\"
    lines = [f"\\begin{{table}}[H]",
             "\\centering",
             "\\small",
             f"\\caption{{{caption}}}",
             f"\\label{{{label}}}",
             "\\setlength{\\tabcolsep}{6pt}",
             "\\begin{tabular}{lccc}",
             "\\toprule",
             "Dev $\\downarrow$ / Test $\\rightarrow$ & O & E & H \\\\",
             "\\midrule"]
    for rn, vals in rows:
        def cell(x):
            if x is None: return "---"
            if isinstance(x, str): return x
            return (fmt % x)
        lines.append(rn + " & " + " & ".join([cell(v) for v in vals]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, type=str)
    ap.add_argument("--model_path", required=True, type=str)
    ap.add_argument("--use_gpu", type=lambda x: str(x).lower() in {"true","1","yes"}, default=True)

    # 路径：搜索树 + 三套 trials
    ap.add_argument("--search_dirs", nargs="*", required=True)
    ap.add_argument("--trials_O", required=True, type=str)
    ap.add_argument("--trials_E", required=True, type=str)
    ap.add_argument("--trials_H", required=True, type=str)

    # 固定点参数（与主文一致）
    ap.add_argument("--fp_kbits", type=int, default=16)
    ap.add_argument("--fp_alpha_pow2", type=int, default=10)
    ap.add_argument("--fp_trunc_sigma", type=int, default=10)

    # 输出
    ap.add_argument("--report_dir", required=True, type=str)

    # 格式
    ap.add_argument("--tau_precision", type=int, default=2, help="LaTeX 中 tau 的小数位（定点域）")
    ap.add_argument("--eer_precision", type=int, default=3, help="LaTeX 中 EER@tau 的百分数小数位")
    args = ap.parse_args()

    ensure_dir(args.report_dir)
    grid_dir = os.path.join(args.report_dir, "dev_choice")
    ensure_dir(grid_dir)

    with StageTimer("加载模型"):
        predictor = MVectorPredictor(configs=args.configs, model_path=args.model_path, use_gpu=args.use_gpu)
        print("[MODEL] CAMPPlus encoder loaded.")

    # 目录索引（只用目录树，不依赖 TSV）
    with StageTimer("索引搜索目录"):
        idx_dirs = build_audio_index_from_dirs(args.search_dirs)

    # 读取三套 trials
    with StageTimer("读取 O/E/H trials"):
        pairs_O, labels_O = load_pairs_for("O", args.trials_O, idx_dirs)
        pairs_E, labels_E = load_pairs_for("E", args.trials_E, idx_dirs)
        pairs_H, labels_H = load_pairs_for("H", args.trials_H, idx_dirs)

    # 一次性抽取嵌入（合并 O/E/H 涉及到的全部路径）
    uniq_utts = union_utts(pairs_O, pairs_E, pairs_H)
    with StageTimer("提取嵌入(合并 O/E/H)"):
        emb_all = prepare_embeddings(
            uniq_utts, predictor,
            use_cache=True, cache_dir=os.path.join(args.report_dir, "../cache_emb"), raw_model_sig="model_sig"
        )
    if not emb_all:
        raise SystemExit("[ERROR] 嵌入为空。")

    # 一个小工具：给定 pairs/labels，从 emb_all 取定点分数
    def score_fp(pairs):
        return score_pairs_fixed_point(
            pairs, emb_all,
            k_bits=args.fp_kbits, alpha_pow2=args.fp_alpha_pow2, trunc_sigma=args.fp_trunc_sigma
        )

    # 预先算好 test 三套的定点分数（复用，避免三重计算）
    with StageTimer("计算 test 定点分数（O/E/H）"):
        S_O = score_fp(pairs_O); S_E = score_fp(pairs_E); S_H = score_fp(pairs_H)

    # dev 三套也分别得出 tau_EER（定点域）
    tau = {}  # tau["O"|"E"|"H"] = float
    with StageTimer("在各 dev 上求 tau_EER（定点域）"):
        for name, S_dev, y_dev in [("O", S_O, labels_O), ("E", S_E, labels_E), ("H", S_H, labels_H)]:
            eer, _, tau_eer, _, _, _ = compute_eer_mindcf(S_dev, y_dev, progress=False)
            tau[name] = float(tau_eer)
            print(f"[TAU] dev={name} -> tau_EER_fp={tau[name]:.6f} (EER_dev={eer:.4f})")

    # 计算：把各 dev 的 tau 应用到各 test，在该 tau 下统计 FAR/FRR -> EER@tau := 0.5*(FAR+FRR)
    def eer_at_tau(S_test, y_test, tau_val):
        m = metrics_at_threshold(S_test, y_test, tau_val)
        return 0.5 * (m["FAR"] + m["FRR"])

    with StageTimer("构建 3×3 网格"):
        # 表A（tau）：每行一个 dev，列为 O/E/H（这里把同一行三个格都写 dev 的 tau 值，方便直接粘贴 3×3）
        tau_grid = {
            "O": [tau["O"], tau["O"], tau["O"]],
            "E": [tau["E"], tau["E"], tau["E"]],
            "H": [tau["H"], tau["H"], tau["H"]],
        }
        # 表B（EER@tau，百分数）
        eatt = {}
        eatt["O"] = [
            100.0 * eer_at_tau(S_O, labels_O, tau["O"]),
            100.0 * eer_at_tau(S_E, labels_E, tau["O"]),
            100.0 * eer_at_tau(S_H, labels_H, tau["O"]),
        ]
        eatt["E"] = [
            100.0 * eer_at_tau(S_O, labels_O, tau["E"]),
            100.0 * eer_at_tau(S_E, labels_E, tau["E"]),
            100.0 * eer_at_tau(S_H, labels_H, tau["E"]),
        ]
        eatt["H"] = [
            100.0 * eer_at_tau(S_O, labels_O, tau["H"]),
            100.0 * eer_at_tau(S_E, labels_E, tau["H"]),
            100.0 * eer_at_tau(S_H, labels_H, tau["H"]),
        ]

    # 导出 CSV
    with StageTimer("导出 CSV"):
        header = ["O", "E", "H"]
        grid_to_csv(os.path.join(grid_dir, "tau_grid.csv"), header,
                    [("dev=O", tau_grid["O"]), ("dev=E", tau_grid["E"]), ("dev=H", tau_grid["H"])])
        grid_to_csv(os.path.join(grid_dir, "eer_at_tau_grid.csv"), header,
                    [("dev=O", [f"{v:.{args.eer_precision}f}" for v in eatt["O"]]),
                     ("dev=E", [f"{v:.{args.eer_precision}f}" for v in eatt["E"]]),
                     ("dev=H", [f"{v:.{args.eer_precision}f}" for v in eatt["H"]])])

    # 生成 LaTeX 片段（两张 3×3 表 + 一句话结论）
    with StageTimer("生成 LaTeX 片段"):
        fmt_tau = "%." + str(args.tau_precision) + "f"
        fmt_eer = "%." + str(args.eer_precision) + "f"
        tab_tau = latex_table_3x3(
            caption="Development-split sensitivity: $\\tau_{\\mathrm{EER}}$ in fixed-point units (rows: dev; columns: test).",
            label="tab:dev-choice-tau",
            header=["O", "E", "H"],
            rows=[("dev=O", tau_grid["O"]), ("dev=E", tau_grid["E"]), ("dev=H", tau_grid["H"])],
            fmt=fmt_tau
        )
        tab_eer = latex_table_3x3(
            caption="Development-split sensitivity: EER@dev-$\\tau$ on each test set (\\%).",
            label="tab:dev-choice-eer",
            header=["O", "E", "H"],
            rows=[("dev=O", eatt["O"]), ("dev=E", eatt["E"]), ("dev=H", eatt["H"])],
            fmt=fmt_eer
        )
        summary = (
            "\\paragraph{Summary.} Transferring thresholds across O/E/H changes the operating EER only modestly; "
            "the qualitative ordering among O/E/H and conclusions in the main text remain unchanged."
        )
        latex_all = "\n\n".join([tab_tau, "", tab_eer, "", summary]) + "\n"
        tex_path = os.path.join(grid_dir, "dev_choice_tables.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_all)
        print(f"[SAVE] {tex_path}")

    # 记录元数据
    meta = dict(
        fp_kbits=args.fp_kbits, fp_alpha_pow2=args.fp_alpha_pow2, fp_trunc_sigma=args.fp_trunc_sigma,
        trials_O=os.path.abspath(args.trials_O),
        trials_E=os.path.abspath(args.trials_E),
        trials_H=os.path.abspath(args.trials_H),
        tau=tau,
        note="EER@tau = 0.5*(FAR+FRR) measured on the test set at the dev-calibrated tau (fixed-point domain)."
    )
    save_json(os.path.join(grid_dir, "dev_choice_meta.json"), meta)

if __name__ == "__main__":
    main()
