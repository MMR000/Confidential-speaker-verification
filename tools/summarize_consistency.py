#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, csv, math, glob

ROOT = sys.argv[1] if len(sys.argv) > 1 else "output"

def load_json(p):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def get_first(d, keys, default=None):
    for k in keys:
        cur = d
        ok = True
        for p in k.split("."):
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False; break
        if ok: return cur
    return default

def as_num(x):
    try:
        if x is None: return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def try_pairs(m, m2):
    # m: consistency_latency.json, m2: metrics_at_thresholds.json (optional)
    cands = [
        "n_pairs","pairs","num_pairs","total_pairs",
        "counts.pairs","counts.kept","kept","n_kept",
        "num_trials","total_trials","n_trials","n_test_pairs","n"
    ]
    v = get_first(m, cands)
    if v is None and isinstance(m.get("counts", {}), dict):
        # 有时候写成 counts = {"pairs": N, ...}
        v = m["counts"].get("pairs") or m["counts"].get("kept")
    if v is None and m2:
        v = get_first(m2, ["n_pairs","pairs","num_pairs","total_pairs","kept","counts.pairs"])
    try:
        return int(v)
    except Exception:
        return 0

def pick_accs(m):
    fp_fp16  = as_num(get_first(m, [
        "consistency.float_vs_fixed","consistency.cosine_vs_fixed",
        "consistency.float32_vs_fixed","consistency.fp_vs_fp16"]))
    fp_mpc   = as_num(get_first(m, [
        "consistency.float_vs_mpc","consistency.cosine_vs_mpc",
        "consistency.float32_vs_mpc","consistency.fp_vs_mpc"]))
    fp16_mpc = as_num(get_first(m, [
        "consistency.fixed_vs_mpc","consistency.fp16_vs_mpc","consistency.fixedpoint_vs_mpc"]))
    three    = as_num(get_first(m, [
        "consistency.three_way","consistency.3way","consistency.all_three"]))
    return fp_fp16, fp_mpc, fp16_mpc, three

def pick_latency(m):
    p50 = as_num(get_first(m, ["latency_ms.p50","latency.p50_ms","p50_ms","latency.p50"]))
    p90 = as_num(get_first(m, ["latency_ms.p90","latency.p90_ms","p90_ms","latency.p90"]))
    p99 = as_num(get_first(m, ["latency_ms.p99","latency.p99_ms","p99_ms","latency.p99"]))
    return p50, p90, p99

def pick_comm(m):
    up   = as_num(get_first(m, ["comm_bytes.uplink","communication.uplink_bytes","uplink_bytes","comm.uplink"]))
    down = as_num(get_first(m, ["comm_bytes.downlink","communication.downlink_bytes","downlink_bytes","comm.downlink"]))
    dim  = as_num(get_first(m, ["comm_bytes.dim","communication.dim","dim","fixed_point.dim"]))
    k    = as_num(get_first(m, ["comm_bytes.k_bits","communication.k_bits","k_bits","fixed_point.k_bits"]))
    return up, down, dim, k

# -------- scan all result dirs --------
dirs = sorted(glob.glob(os.path.join(ROOT, "consistency_*")))
rows = []
missing_files = []
for d in dirs:
    jl = os.path.join(d, "consistency_latency.json")
    if not os.path.isfile(jl):
        missing_files.append((d, jl)); continue
    jm = os.path.join(d, "metrics_at_thresholds.json")
    m = load_json(jl)
    m2 = load_json(jm) if os.path.isfile(jm) else {}

    name = os.path.basename(d)
    pairs = try_pairs(m, m2)
    fp_fp16, fp_mpc, fp16_mpc, three = pick_accs(m)
    p50, p90, p99 = pick_latency(m)
    up, down, dim, kbits = pick_comm(m)
    rows.append([name, pairs, fp_fp16, fp_mpc, fp16_mpc, three, p50, p90, p99, up, down, dim, kbits])

# pretty print
def pct(x):  return "nan" if (x is None or (isinstance(x,float) and math.isnan(x))) else f"{100.0*float(x):.2f}"
def num(x):  return "nan" if (x is None or (isinstance(x,float) and math.isnan(x))) else f"{float(x):.1f}"
def iby(x):  return "nan" if (x is None or (isinstance(x,float) and math.isnan(x))) else f"{int(round(float(x)))}"

if not rows and not missing_files:
    print("[WARN] 没有找到任何 consistency_* 目录。路径：", ROOT)

print("{:<34s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}   {:>6s}/{:>6s}/{:>6s}   {:>9s} {:>9s}".format(
    "dir","pairs","fp-fp16","fp-mpc","fp16-mpc","3way","p50","p90","p99","uplink(B)","down(B)"))
for r in rows:
    print("{:<34s} {:>8d} {:>8s} {:>8s} {:>8s} {:>8s}   {:>6s}/{:>6s}/{:>6s}   {:>9s} {:>9s}".format(
        r[0], int(r[1]), pct(r[2]), pct(r[3]), pct(r[4]), pct(r[5]),
        num(r[6]), num(r[7]), num(r[8]), iby(r[9]), iby(r[10])
    ))

# CSV
csv_path = os.path.join(ROOT, "consistency_summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dir","pairs","fp_vs_fp16","fp_vs_mpc","fp16_vs_mpc","three_way",
                "p50_ms","p90_ms","p99_ms","uplink_bytes","downlink_bytes","dim","k_bits"])
    for r in rows:
        w.writerow(r)
print(f"[SAVE] {csv_path}")

# debug hints for any row with pairs==0
for r in rows:
    if int(r[1]) == 0:
        jl = os.path.join(ROOT, r[0], "consistency_latency.json")
        m = load_json(jl)
        print(f"[DEBUG] {r[0]} 顶层可用键：", ", ".join(sorted(m.keys())))
        if "counts" in m and isinstance(m["counts"], dict):
            print(f"[DEBUG] {r[0]} counts.* 键：", ", ".join(sorted(m["counts"].keys())))

for d, jl in missing_files:
    print(f"[WARN] 缺少 {jl}")
