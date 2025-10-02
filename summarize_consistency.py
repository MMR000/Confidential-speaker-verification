#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, csv, sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else "output"
SETS = [
    ("O",  os.path.join(ROOT, "consistency_vox1_O_devO_testO", "consistency_latency.json")),
    ("E",  os.path.join(ROOT, "consistency_vox1_E_devO_testE", "consistency_latency.json")),
    ("H",  os.path.join(ROOT, "consistency_vox1_H_devO_testH", "consistency_latency.json")),
]
rows = []
for name, path in SETS:
    if not os.path.isfile(path):
        print(f"[WARN] missing {name}: {path}")
        continue
    m = json.load(open(path,"r"))
    pairs = m["n_pairs"]
    acc_fp_fp16  = m["consistency"]["float_vs_fixed"]
    acc_fp_mpc   = m["consistency"]["float_vs_mpc"]
    acc_fp16_mpc = m["consistency"]["fixed_vs_mpc"]
    acc_3way     = m["consistency"]["three_way"]
    p50 = m["latency_ms"]["p50"]; p90 = m["latency_ms"]["p90"]; p99 = m["latency_ms"]["p99"]
    up = m["comm_bytes"]["uplink"]; down = m["comm_bytes"]["downlink"]
    rows.append([name, pairs, acc_fp_fp16, acc_fp_mpc, acc_fp16_mpc, acc_3way, p50, p90, p99, up, down])
# print pretty
print("{:<3s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}   {:>6s}/{:>6s}/{:>6s}   {:>7s} {:>7s}".format(
    "set","pairs","fp-fp16","fp-mpc","fp16-mpc","3way","p50","p90","p99","uplink","downlink"))
for r in rows:
    print("{:<3s} {:>8d} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}   {:>6.1f}/{:>6.1f}/{:>6.1f}   {:>7.0f} {:>7.0f}".format(
        r[0], r[1], 100*r[2], 100*r[3], 100*r[4], 100*r[5], r[6], r[7], r[8], r[9], r[10]))
# dump CSV for paper
csv_path = os.path.join(ROOT, "consistency_summary.csv")
with open(csv_path,"w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["set","pairs","fp_vs_fp16","fp_vs_mpc","fp16_vs_mpc","three_way","p50_ms","p90_ms","p99_ms","uplink_bytes","downlink_bytes"])
    for r in rows: w.writerow(r)
print(f"[SAVE] {csv_path}")

