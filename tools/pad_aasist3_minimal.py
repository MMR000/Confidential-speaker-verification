#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小可用版：AASIST3 PAD 推理（修正版）
- 需要提供 --code_dir 指向本地 AASIST3 源码目录（里面应有 model.py 或包能导入 aasist3）
- 自动从 Hugging Face 拉取 MTUCI/AASIST3 权重（仅权重，无源码）
- 读 trials（1 列 path 或 2 列 label\tpath），写出“三列 TSV：score\tlabel(optional)\tpath”
  注：score 为 “spoof 概率”，越大越像伪造
"""

import os, sys, argparse, csv
from typing import List, Tuple
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import snapshot_download

SAMPLE_RATE = 16000
TARGET_LEN = 64600  # 约 4 秒 / 16k

def _try_import_aasist3(code_dir: str):
    """
    尝试从给定源码目录导入 aasist3 模块。
    支持几种常见布局：
      - from model import aasist3
      - from aasist3 import aasist3
      - import model; model.aasist3
    """
    if not code_dir or not os.path.isdir(code_dir):
        raise RuntimeError(f"--code_dir 目录不存在：{code_dir}")

    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    last_err = None
    for stmt in [
        "from model import aasist3 as A",
        "from aasist3 import aasist3 as A",
        "import model as A",  # 若 model.py 里暴露 AASIST3 接口
    ]:
        try:
            ns = {}
            exec(stmt, ns, ns)
            A = ns["A"]
            return A
        except Exception as e:
            last_err = e
            continue

    # 打印一下目录里都有什么，帮助定位
    listing = "\n  - " + "\n  - ".join(sorted(os.listdir(code_dir))) if os.path.isdir(code_dir) else ""
    raise RuntimeError(
        "无法从 --code_dir 导入 AASIST3 接口。\n"
        f"尝试导入失败：{last_err}\n"
        f"请确认源码目录包含 model.py 或包能 `from model import aasist3` / `from aasist3 import aasist3`。\n"
        f"目录内容：{listing}\n"
    )

def load_aasist3(repo_id: str, code_dir: str, device: str = "cuda"):
    # 1) 下载权重快照（仅权重+config，无源码）
    weights_dir = snapshot_download(repo_id)  # MTUCI/AASIST3
    # 2) 导入源码里的 aasist3 工厂
    aasist3_mod = _try_import_aasist3(code_dir)
    # 3) from_pretrained 用本地权重目录
    model = aasist3_mod.from_pretrained(weights_dir)
    model.eval()
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(dev)
    return model, dev

def read_trials(trials_path: str) -> List[Tuple[str, str]]:
    """
    返回列表 [(label_or_empty, path)]；label 可为空字符串
    兼容：
      - 1列：path
      - 2列：label, path
      - 多列：默认取最后一列为 path；第一列若像标签则作 label
    """
    rows = []
    with open(trials_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for r in reader:
            if not r: continue
            r = [x.strip() for x in r if x is not None]
            if len(r) == 1:
                rows.append(("", r[0]))
            else:
                first = r[0].lower()
                is_label_like = first in {"0","1","bonafide","spoof","real","synthesized","cloned"}
                label = r[0] if is_label_like else ""
                path = r[-1]
                rows.append((label, path))
    return rows

def wav_to_tensor(path: str, device: torch.device) -> torch.Tensor:
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
    return wav.to(device)

class PADDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], device: torch.device):
        self.pairs = pairs
        self.device = device
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        label, path = self.pairs[idx]
        x = wav_to_tensor(path, self.device)
        if isinstance(label, str) and label != "":
            l = label.lower()
            if l in {"1","spoof","synthesized","cloned"}:
                label = "1"
            elif l in {"0","bonafide","real"}:
                label = "0"
            else:
                label = ""
        return x, label, path

def _collate(batch):
    xs, labs, paths = zip(*batch)
    if isinstance(xs[0], torch.Tensor):
        xs = torch.stack(xs, dim=0)
    return xs, list(labs), list(paths)

def infer(model, device, dataset: PADDataset, batch_size: int = 16, num_workers: int = 0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_collate)
    out_rows = []
    model.eval()
    with torch.no_grad():
        for xs, labs, paths in loader:
            out = model(xs)  # 期望 logits [B,2]，0=bonafide, 1=spoof
            probs = torch.softmax(out, dim=1)
            spoof_prob = probs[:, 1].detach().cpu().tolist()
            for s, lab, p in zip(spoof_prob, labs, paths):
                out_rows.append((f"{s:.6f}", lab, p))
    return out_rows

def write_scores(out_path: str, rows: List[Tuple[str, str, str]]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["score", "label(optional)", "path"])
        w.writerows(rows)
    print(f"[OK] 写出：{out_path}  共 {len(rows)} 行")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True, help="pad_trials_*.tsv（1列或2列）")
    ap.add_argument("--out", required=True, help="输出 scores：score\\tlabel(optional)\\tpath")
    ap.add_argument("--repo", default="MTUCI/AASIST3", help="HF 权重仓（默认 MTUCI/AASIST3）")
    ap.add_argument("--code_dir", required=True, help="本地 AASIST3 源码目录（必须包含可导入的 aasist3）")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    model, device = load_aasist3(args.repo, args.code_dir, args.device)
    pairs = read_trials(args.trials)
    ds = PADDataset(pairs, device)
    rows = infer(model, device, ds, batch_size=args.batch_size)
    write_scores(args.out, rows)

if __name__ == "__main__":
    main()
