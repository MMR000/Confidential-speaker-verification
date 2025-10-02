#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ppsv_consistency.py
- 三方一致率：float32 vs fixed-point vs MPC（Crypten）
- O/E/H 各集合的决策一致率（不足 5 万自动补 trials）
- 并列阈值处理：统一使用 decision = (score >= tau)
- 延迟 CDF（p50/p90/p99）与每次验证的上下行字节数估计
- 直接复用现有 ppsv_runner.py 中的大量实现（嵌入提取/打分/固定点/MPC/作图等）

依赖：
  pip install -U numpy tqdm matplotlib crypten torch
  （其余依赖与 ppsv_runner 相同）

用法见本文末尾 shell 命令示例。
"""
import os, sys, json, math, time, argparse, random
import numpy as np

# ---- 引入你已有的“一键评测器”工具集 -----------------------------------------
# 确保本文件与 ppsv_runner.py 位于同一项目内
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ppsv_runner import (
    MVectorPredictor, ensure_dir, StageTimer, HAS_TQDM, pb_iter,
    build_audio_index_from_dirs, build_audio_index_from_validated,
    load_trials_with_index, auto_generate_trials, save_trials,
    prepare_embeddings, score_pairs_from_emb,
    score_pairs_fixed_point, compute_eer_mindcf, find_threshold_for_far,
    metrics_at_threshold, plot_latency_cdf, plot_comm_bar,
    estimate_global_precision, fit_lda_transform, apply_salt
)

# -----------------------------
# 工具
# -----------------------------
def log(s): print(s, flush=True)

def pct(a): return float(a*100.0)

def percentile(arr, p):
    if len(arr)==0: return float("nan")
    return float(np.percentile(np.asarray(arr, dtype=np.float64), p))

def fmt_bytes(n):
    units=["B","KB","MB","GB"]
    x=float(n); i=0
    while x>=1024 and i+1<len(units):
        x/=1024; i+=1
    return f"{x:.2f}{units[i]}"

# -----------------------------
# trials 補齐（针对 O 集合 < 50k 的情况）
# -----------------------------
def topup_trials_if_needed(pairs, labels, target_min, spk2utts, seed=123):
    """若 pairs 数量不足 target_min，则从 spk2utts 自动补齐不重叠 trials。"""
    if len(pairs) >= target_min:
        return pairs, labels, 0
    random.seed(seed)
    needed = target_min - len(pairs)
    # 粗略保留原始正负比例
    pos_ratio = (sum(labels)/len(labels)) if labels else 0.5
    pos_need = int(round(needed * pos_ratio))
    neg_need = needed - pos_need

    # 已有集合避免重复
    have = set()
    for a,b in pairs:
        key = (a,b) if a<b else (b,a)
        have.add(key)

    # 生成尽量不重复的
    add_pairs, add_labels = [], []
    speakers = [s for s in spk2utts.keys() if len(spk2utts[s])>=1]
    speakers_2 = [s for s in spk2utts.keys() if len(spk2utts[s])>=2]
    # 正样本
    random.shuffle(speakers_2)
    for s in speakers_2:
        if pos_need<=0: break
        utts = list(spk2utts[s]); random.shuffle(utts)
        for i in range(len(utts)):
            for j in range(i+1,len(utts)):
                a,b = utts[i],utts[j]
                k=(a,b) if a<b else (b,a)
                if k in have: continue
                have.add(k); add_pairs.append((a,b)); add_labels.append(1); pos_need-=1
                if pos_need<=0: break
            if pos_need<=0: break
    # 负样本
    spk_list = list(speakers)
    while neg_need>0 and len(spk_list)>=2:
        s1,s2 = random.sample(spk_list, 2)
        a = random.choice(spk2utts[s1]); b = random.choice(spk2utts[s2])
        k=(a,b) if a<b else (b,a)
        if k in have: continue
        have.add(k); add_pairs.append((a,b)); add_labels.append(0); neg_need-=1

    pairs2 = pairs + add_pairs
    labels2 = labels + add_labels
    return pairs2, labels2, len(add_pairs)

# -----------------------------
# 明文( float32 ) 决策
# -----------------------------
def decide_float32(pairs, emb, tau, backend="cosine"):
    S = score_pairs_from_emb(pairs, emb, mode=("euclid" if backend=="euclid" else "cosine"))
    # 并列阈值：统一 >= tau
    dec = (S >= tau).astype(np.int32)
    return S, dec

# -----------------------------
# Fixed-point 决策（使用 dev 校准阈值或指定阈值）
# -----------------------------
def decide_fixed_point(pairs, emb, tau_fp, k_bits=16, alpha_pow2=10, trunc_sigma=10):
    Sfp = score_pairs_fixed_point(pairs, emb, k_bits=k_bits, alpha_pow2=alpha_pow2, trunc_sigma=trunc_sigma)
    dec = (Sfp >= tau_fp).astype(np.int32)
    return Sfp, dec

# -----------------------------
# MPC 决策（Crypten 逐对；返回决策与时延）
# -----------------------------
def decide_mpc(pairs, emb, tau, backend="crypten"):
    times = []
    decisions = []
    if backend=="local":
        it = pb_iter(pairs, desc="mpc-local", total=len(pairs)) if HAS_TQDM else pairs
        for (a,b) in it:
            t0=time.time()
            s = float(np.dot(emb[a], emb[b]))
            d = int(s >= tau)
            times.append(time.time()-t0); decisions.append(d)
        return np.asarray(decisions, np.int32), np.asarray(times, np.float64)

    # crypten
    try:
        import torch, crypten
        crypten.init()
    except Exception as e:
        raise RuntimeError(f"[MPC] 需要 crypten/torch：{e}")

    it = pb_iter(pairs, desc="mpc-crypten", total=len(pairs)) if HAS_TQDM else pairs
    import torch, crypten
    tt = torch.tensor([tau], dtype=torch.float32)
    ct = crypten.cryptensor(tt)
    for (a,b) in it:
        t0=time.time()
        tx = torch.from_numpy(emb[a]).float()
        ty = torch.from_numpy(emb[b]).float()
        cx = crypten.cryptensor(tx); cy = crypten.cryptensor(ty)
        cscore = (cx*cy).sum()
        c_ge = cscore.ge(ct)
        same = bool(c_ge.get_plain_text().item() > 0.5)
        decisions.append(1 if same else 0)
        times.append(time.time()-t0)
    return np.asarray(decisions, np.int32), np.asarray(times, np.float64)

# -----------------------------
# 通信开销估计（协议负载下界）
# -----------------------------
def comm_bytes_per_verification(dim, k_bits=16, scheme="2pc_shares"):
    """
    返回 (uplink_bytes, downlink_bytes) 的理论下界估计。
    假设：
      - 每方将其定点量化后的向量做加法秘密共享（2 份 share），各发送 1 份给对方；
      - 仅比较结果返回 1 bit（向上取 1 字节）；
    则每次验证：
      uplink ≈ (k_bits/8)*dim
      downlink ≈ (k_bits/8)*dim + 1（约等于 uplink）
    注：实际 MPC 框架会有额外协议开销/握手/激活函数协议等，这里给“协议负载下界”用于对比。
    """
    base = (k_bits/8.0) * dim
    up = int(math.ceil(base))
    down = int(math.ceil(base)) + 1
    return up, down

# -----------------------------
# 主流程
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    # 模型/数据入口
    ap.add_argument("--configs", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--use_gpu", type=lambda x: str(x).lower() in {"true","1","yes"}, default=True)

    ap.add_argument("--search_dirs", nargs="*", required=True)
    ap.add_argument("--dev_search_dirs", nargs="*", required=True)
    ap.add_argument("--dev_trials", type=str, required=True)
    ap.add_argument("--trials", type=str, required=True)

    # 输出
    ap.add_argument("--report_dir", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, default="output/cache_emb")
    ap.add_argument("--use_cache", type=lambda x: str(x).lower() in {"true","1","yes"}, default=True)

    # backend / 阈值策略
    ap.add_argument("--backend", type=str, default="cosine", choices=["cosine","euclid"])
    ap.add_argument("--auto_tau_from_dev", action="store_true", help="用 dev 集 float32 的 EER 阈值作为明文/MPC阈值")
    ap.add_argument("--threshold", type=float, default=0.60, help="若未 --auto_tau_from_dev 时用于明文/MPC 的阈值")

    # fixed-point 参数
    ap.add_argument("--fp_kbits", type=int, default=16)
    ap.add_argument("--fp_alpha_pow2", type=int, default=10)
    ap.add_argument("--fp_trunc_sigma", type=int, default=10)

    # 至少 N 对（O/E/H 各集合）
    ap.add_argument("--min_pairs", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)

    # 运行 MPC（crypten）与延迟统计
    ap.add_argument("--run_mpc", action="store_true")
    ap.add_argument("--plot_cdf", action="store_true")
    ap.add_argument("--plot_comm_bar", action="store_true")

    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    ensure_dir(args.report_dir); ensure_dir(args.cache_dir)

    with StageTimer("加载模型"):
        predictor = MVectorPredictor(configs=args.configs, model_path=args.model_path, use_gpu=args.use_gpu)
        log("[MODEL] 成功加载嵌入模型。")

    # ---------- 索引 test/dev ----------
    with StageTimer("索引(test/dev)"):
        idx_dirs = build_audio_index_from_dirs(args.search_dirs)
        dev_idx_dirs = build_audio_index_from_dirs(args.dev_search_dirs)
        # 这里不强制 validated.tsv，项目用目录索引即可
        spk2utts_test = {}
        for k,v in idx_dirs.items():
            p=v
            if not any(p.lower().endswith(x) for x in [".wav",".flac",".mp3",".m4a",".ogg"]):
                continue
            spk = os.path.basename(os.path.dirname(os.path.dirname(p))) if "/id" in p.replace("\\","/") \
                  else os.path.basename(os.path.dirname(p))
            spk2utts_test.setdefault(spk, []).append(p)
        spk2utts_dev = {}
        for k,v in dev_idx_dirs.items():
            p=v
            if not any(p.lower().endswith(x) for x in [".wav",".flac",".mp3",".m4a",".ogg"]):
                continue
            spk = os.path.basename(os.path.dirname(os.path.dirname(p))) if "/id" in p.replace("\\","/") \
                  else os.path.basename(os.path.dirname(p))
            spk2utts_dev.setdefault(spk, []).append(p)
        log(f"[INDEX] test speakers={len(spk2utts_test)} dev speakers={len(spk2utts_dev)}")

    # ---------- 读取 trials ----------
    with StageTimer("读取 trials"):
        index_all = dict(idx_dirs)
        test_pairs, test_labels = load_trials_with_index(args.trials, index=index_all, base_hint=os.path.dirname(args.trials))
        dev_pairs,  dev_labels  = load_trials_with_index(args.dev_trials, index=dict(dev_idx_dirs), base_hint=os.path.dirname(args.dev_trials))

        if not test_pairs:
            raise SystemExit("[ERROR] test trials 解析为空。")
        if not dev_pairs or not dev_labels:
            raise SystemExit("[ERROR] dev trials 需要用于校准阈值与 fixed-point。")

        # 若不足 min_pairs 则自动补齐
        if args.min_pairs>0 and len(test_pairs)<args.min_pairs:
            test_pairs, test_labels, added = topup_trials_if_needed(test_pairs, test_labels, args.min_pairs, spk2utts_test, seed=args.seed)
            if added>0:
                out_topup = os.path.join(args.report_dir, "autogen_topup_trials.txt")
                save_trials(out_topup, test_pairs, test_labels)
                log(f"[TRIALS] 原官方 trials={len(test_pairs)-added}，自动补齐={added} -> 总计={len(test_pairs)}")

    # ---------- 嵌入提取（去重） ----------
    uniq_test_utts = sorted({p for ab in test_pairs for p in ab})
    uniq_dev_utts  = sorted({p for ab in dev_pairs  for p in ab})
    with StageTimer("提取嵌入(dev/test)"):
        emb_test = prepare_embeddings(uniq_test_utts, predictor,
                                      use_cache=args.use_cache, cache_dir=args.cache_dir, raw_model_sig="model_sig")
        emb_dev  = prepare_embeddings(uniq_dev_utts,  predictor,
                                      use_cache=args.use_cache, cache_dir=args.cache_dir, raw_model_sig="model_sig")
    if not emb_test or not emb_dev:
        raise SystemExit("[ERROR] 嵌入提取失败。")

    # ---------- dev 上求明文阈值（EER 或给定） ----------
    with StageTimer("阈值校准(dev)"):
        S_dev_float, _ = decide_float32(dev_pairs, emb_dev, tau=0.0, backend=args.backend)
        eer_dev, _, tau_eer_dev, _, _, _ = compute_eer_mindcf(S_dev_float, np.asarray(dev_labels, np.int32))
        if args.auto_tau_from_dev:
            tau_float = float(tau_eer_dev)
        else:
            tau_float = float(args.threshold)
        log(f"[TAU] dev: tau_eer={tau_eer_dev:.6f}；用于明文/MPC 的阈值 tau_float={tau_float:.6f}")

        # fixed-point 阈值来自 dev 的定点域
        S_dev_fp = score_pairs_fixed_point(dev_pairs, emb_dev,
                                           k_bits=args.fp_kbits, alpha_pow2=args.fp_alpha_pow2, trunc_sigma=args.fp_trunc_sigma)
        _, _, tau_eer_dev_fp, _, _, _ = compute_eer_mindcf(S_dev_fp, np.asarray(dev_labels, np.int32))
        tau_fp = float(tau_eer_dev_fp)
        log(f"[TAU] dev: tau_eer_fp={tau_fp:.6f}（定点域）")

    # ---------- test 上三路决策 ----------
    with StageTimer("三路决策(test)"):
        S_float, D_float = decide_float32(test_pairs, emb_test, tau=tau_float, backend=args.backend)
        S_fp,    D_fp    = decide_fixed_point(test_pairs, emb_test, tau_fp, k_bits=args.fp_kbits,
                                              alpha_pow2=args.fp_alpha_pow2, trunc_sigma=args.fp_trunc_sigma)
        if args.run_mpc:
            D_mpc, mpc_times = decide_mpc(test_pairs, emb_test, tau=tau_float, backend="crypten")
        else:
            # 若不跑 MPC，就用本地 float 代替（仅用于一致率格式打通）
            D_mpc, mpc_times = D_float.copy(), np.zeros(len(test_pairs), dtype=np.float64)

    # ---------- 一致率统计 ----------
    with StageTimer("一致率统计"):
        n = len(test_pairs)
        same_f_fp  = int(np.sum(D_float == D_fp))
        same_f_mpc = int(np.sum(D_float == D_mpc))
        same_fp_m  = int(np.sum(D_fp    == D_mpc))
        all_three  = int(np.sum((D_float == D_fp) & (D_float == D_mpc)))
        res_consistency = dict(
            total_pairs=n,
            agree_float_vs_fixed = same_f_fp,
            agree_float_vs_mpc   = same_f_mpc,
            agree_fixed_vs_mpc   = same_fp_m,
            agree_all_three      = all_three,
            rate_float_vs_fixed  = pct(same_f_fp/n),
            rate_float_vs_mpc    = pct(same_f_mpc/n),
            rate_fixed_vs_mpc    = pct(same_fp_m/n),
            rate_all_three       = pct(all_three/n),
            tau_float=tau_float,
            tau_fixed_point=tau_fp,
            tie_policy=">= tau (统一)"
        )

    # ---------- 延迟与通信 ----------
    with StageTimer("延迟/通信统计"):
        lat = np.asarray(mpc_times, np.float64)
        p50 = percentile(lat*1000, 50)
        p90 = percentile(lat*1000, 90)
        p99 = percentile(lat*1000, 99)
        # 维度由任一嵌入向量得出
        any_vec = next(iter(emb_test.values()))
        dim = int(any_vec.shape[0])
        up, down = comm_bytes_per_verification(dim, k_bits=args.fp_kbits)
        res_latency = dict(
            latency_ms_p50=p50, latency_ms_p90=p90, latency_ms_p99=p99,
            dim=dim, fp_kbits=args.fp_kbits,
            uplink_bytes_per_verification=up,
            downlink_bytes_per_verification=down,
            uplink_human=fmt_bytes(up), downlink_human=fmt_bytes(down)
        )
        # 可选：CDF 图与通信条形图
        if args.plot_cdf and len(lat)>0:
            out_cdf = os.path.join(args.report_dir, "latency_cdf_mpc.png")
            plot_latency_cdf(lat, out_cdf)
        if args.plot_comm_bar:
            out_comm = os.path.join(args.report_dir, "comm_bar.png")
            plot_comm_bar(dim, out_comm, k_bits=args.fp_kbits)

    # ---------- 保存结果 ----------
    out_json = {
        "consistency": res_consistency,
        "latency_comm": res_latency,
        "backend": args.backend,
        "min_pairs_target": args.min_pairs,
        "dev_trials": os.path.basename(args.dev_trials),
        "test_trials": os.path.basename(args.trials),
        "notes": "一致率基于并列阈值 >= tau；通信为协议负载下界估计。"
    }
    ensure_dir(args.report_dir)
    with open(os.path.join(args.report_dir, "consistency_latency.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    log(f"[SAVE] {os.path.join(args.report_dir, 'consistency_latency.json')}")

    # 追加：若有标签，输出阈值点处指标供复核
    if test_labels:
        y = np.asarray(test_labels, np.int32)
        pack = lambda S, tau: metrics_at_threshold(S, y, tau)
        summary = dict(
            float32_at_tau=pack(S_float, tau_float),
            fixed_point_at_tau=pack(S_fp, tau_fp)
        )
        with open(os.path.join(args.report_dir, "metrics_at_thresholds.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log(f"[SAVE] {os.path.join(args.report_dir, 'metrics_at_thresholds.json')}")

if __name__ == "__main__":
    main()

