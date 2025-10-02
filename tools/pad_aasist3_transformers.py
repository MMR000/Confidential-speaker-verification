#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
尝试用 HuggingFace Transformers 直接加载 MTUCI/AASIST3 并做 PAD 推理。
成功：输出三列TSV：score\tlabel(optional)\tpath   （score=spoof概率）
失败：明确报错并提示改用方案B。
"""
import os, sys, argparse, csv, torch, torchaudio
from typing import List, Tuple
from transformers import AutoProcessor, AutoModelForAudioClassification
from huggingface_hub import hf_hub_download

SAMPLE_RATE = 16000
TARGET_LEN = 64600

def read_trials(trials_path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(trials_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            parts = [p.strip() for p in line.strip().split("\t") if p is not None]
            if len(parts) == 1:
                rows.append(("", parts[0]))
            else:
                first = parts[0].lower()
                is_label = first in {"0","1","bonafide","spoof","real","synthesized","cloned"}
                label = parts[0] if is_label else ""
                path = parts[-1]
                rows.append((label, path))
    return rows

def load_wav(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    T = wav.shape[1]
    if T < TARGET_LEN:
        wav = torch.nn.functional.pad(wav, (0, TARGET_LEN - T))
    elif T > TARGET_LEN:
        wav = wav[:, :TARGET_LEN]
    return wav.squeeze(0)  # [T]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="MTUCI/AASIST3")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    dev = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    # 先探测仓库是否至少可下载文件（避免401/404）
    try:
        _ = hf_hub_download(args.repo, filename="config.json")
    except Exception as e:
        print(f"[ERR] 无法访问仓库 {args.repo} ：{e}")
        sys.exit(2)

    # 尝试AutoProcessor/AutoModel加载
    try:
        processor = AutoProcessor.from_pretrained(args.repo)
        model = AutoModelForAudioClassification.from_pretrained(args.repo)
        model.to(dev).eval()
    except Exception as e:
        print("[ERR] Transformers 自动加载失败：", e)
        print(">> 该仓库可能不是标准 Transformers 模型。请改用方案B（见说明）。")
        sys.exit(3)

    pairs = read_trials(args.trials)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["score", "label(optional)", "path"])
        with torch.no_grad():
            buf = []
            meta = []
            for lab, p in pairs:
                wav = load_wav(p)
                buf.append(wav)
                meta.append((lab, p))
                if len(buf) == args.batch:
                    inputs = processor(buf, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
                    inputs = {k: v.to(dev) for k,v in inputs.items()}
                    logits = model(**inputs).logits  # [B, C]
                    probs = torch.softmax(logits, dim=-1)
                    if probs.shape[1] == 2:
                        spoof_prob = probs[:, 1].detach().cpu().tolist()
                    else:
                        # 兜底：若标签未知，取“最大类≠bonafide”的概率；这里简单用1-argmax为0的概率
                        spoof_prob = (1.0 - probs[:, 0]).detach().cpu().tolist()
                    for s, (lab, pth) in zip(spoof_prob, meta):
                        w.writerow([f"{s:.6f}", lab, pth])
                    buf, meta = [], []
            if buf:
                inputs = processor(buf, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
                inputs = {k: v.to(dev) for k,v in inputs.items()}
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                if probs.shape[1] == 2:
                    spoof_prob = probs[:, 1].detach().cpu().tolist()
                else:
                    spoof_prob = (1.0 - probs[:, 0]).detach().cpu().tolist()
                for s, (lab, pth) in zip(spoof_prob, meta):
                    w.writerow([f"{s:.6f}", lab, pth])

    print(f"[OK] 写出：{args.out}")

if __name__ == "__main__":
    main()
