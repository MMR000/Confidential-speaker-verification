#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AASIST3(KAN) 推理器：读取 PAD trials（单通道路径），批量计算 spoof 概率并输出标准 TSV
依赖：
  pip install huggingface_hub torch torchaudio pandas tqdm
用法：
  python tools/pad_aasist3_infer.py \
    --trials output/pad_trials_val.tsv \
    --out output/report_pad_val_nocloned/scores_aasist3.tsv
"""

import os, sys, argparse, importlib.util
import pandas as pd
import torch, torchaudio
from tqdm import tqdm

# ----------------------
# 工具函数
# ----------------------
def read_trials(trials_path: str):
    """兼容两种常见列名：
       1) label \t path
       2) path \t label
       如 label 为字符串（real/bona/... vs spoof/tts/...），自动映射到 1/0（1=真，0=假）
    """
    df = pd.read_csv(trials_path, sep="\t")
    # 猜列
    if "path" in df.columns:
        path_col = "path"
        label_col = "label" if "label" in df.columns else [c for c in df.columns if c != "path"][0]
    else:
        # 取前两列
        cols = list(df.columns)[:2]
        # 判断哪个像路径列
        if df[cols[0]].astype(str).str.contains(r"[\\/]").mean() >= 0.5:
            path_col, label_col = cols[0], cols[1]
        else:
            path_col, label_col = cols[1], cols[0]

    paths = df[path_col].astype(str).tolist()
    lab = df[label_col]
    if lab.dtype == object:
        y = lab.astype(str).str.lower().map(lambda s: 1 if ("real" in s or "bona" in s) else 0).fillna(0).astype(int)
    else:
        y = lab.astype(int)
    return paths, y.tolist()


def to_mono_16k_pad(audio: torch.Tensor, sr: int, target_len: int = 64600) -> torch.Tensor:
    """转单声道、重采样到16k、裁/补到 target_len（默认~4.04s）"""
    # audio: [C, T] or [T]
    if audio.dim() == 2:
        audio = audio.mean(dim=0)  # -> [T]
    # 重采样
    if sr != 16000:
        audio = torchaudio.functional.resample(audio.unsqueeze(0), sr, 16000).squeeze(0)
    T = audio.shape[-1]
    if T < target_len:
        audio = torch.nn.functional.pad(audio, (0, target_len - T))
    else:
        audio = audio[:target_len]
    return audio  # [T]


def dynamic_import_from_dir(dir_path: str, module_rel: str):
    """从目录动态 import 一个模块（例如 'model/aasist3.py'）"""
    full = os.path.join(dir_path, module_rel)
    mod_name = os.path.splitext(os.path.basename(module_rel))[0]
    spec = importlib.util.spec_from_file_location(mod_name, full)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{full}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def load_aasist3(repo_id: str, device: str, cache_dir: str):
    """使用 huggingface_hub 下载 MTUCI/AASIST3，并加载其 from_pretrained"""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=False)
    # 典型目录里有 model/aasist3.py
    # 允许两种位置：model/aasist3.py 或 aasist3.py
    candidates = ["model/aasist3.py", "aasist3.py"]
    errs = []
    for rel in candidates:
        try:
            if os.path.exists(os.path.join(local_dir, rel)):
                mod = dynamic_import_from_dir(local_dir, rel)
                if hasattr(mod, "aasist3"):
                    cls = getattr(mod, "aasist3")
                elif hasattr(mod, "AASIST3"):
                    cls = getattr(mod, "AASIST3")
                else:
                    raise ImportError("模块中未找到 aasist3/AASIST3 类")
                model = cls.from_pretrained(local_dir)
                model.eval().to(device)
                print(f"[LOAD] AASIST3 ok -> {repo_id} ({rel}) on {device}")
                return model
        except Exception as e:
            errs.append(f"{rel}: {repr(e)}")
    raise RuntimeError("AASIST3 加载失败。\n" + "\n".join(errs))


# ----------------------
# 主流程
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True, help="PAD 评测列表（tsv），至少包含 path 和 label 两列")
    ap.add_argument("--out", required=True, help="输出 TSV 路径（score\tlabel(optional)\tpath）")
    ap.add_argument("--repo", default="MTUCI/AASIST3", help="HuggingFace 模型仓库名")
    ap.add_argument("--batch", type=int, default=16, help="批大小")
    ap.add_argument("--win_len", type=int, default=64600, help="输入帧长（默认 64600≈4.04s/16kHz）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache_dir", default=None, help="huggingface_hub 缓存目录（可留空）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 加载模型
    model = load_aasist3(args.repo, args.device, args.cache_dir)

    # 读 trials
    paths, labels = read_trials(args.trials)

    # 批推理
    rows = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), args.batch), desc="AASIST3 infer"):
            batch_paths = paths[i : i + args.batch]
            batch_labels = labels[i : i + args.batch]
            wavs = []
            for p in batch_paths:
                try:
                    audio, sr = torchaudio.load(p)
                    audio = to_mono_16k_pad(audio, sr, args.win_len)
                    wavs.append(audio)
                except Exception:
                    # 坏样本跳过（保持行数对齐会更复杂；我们这里直接丢弃对应 trial）
                    wavs.append(None)

            # 过滤掉加载失败的
            valid_idx = [k for k, w in enumerate(wavs) if w is not None]
            if not valid_idx:
                continue
            valid_wavs = [wavs[k] for k in valid_idx]
            valid_labels = [batch_labels[k] for k in valid_idx]
            valid_paths = [batch_paths[k] for k in valid_idx]

            x = torch.stack(valid_wavs, dim=0).to(torch.float32).to(args.device)  # [B, T]
            # AASIST3 的 forward 一般输出 logits [B,2]
            logits = model(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = torch.softmax(logits, dim=-1)  # [B,2]
            spoof_prob = probs[:, 1].detach().cpu().tolist()

            for s, y, p in zip(spoof_prob, valid_labels, valid_paths):
                rows.append([s, y, p])

    # 写出
    df_out = pd.DataFrame(rows, columns=["score", "label(optional)", "path"])
    df_out.to_csv(args.out, sep="\t", index=False)
    print(f"[OK] 写出：{args.out}  行数={len(df_out)}")
    print("[NOTE] score 越大 = 越“假/合成”(spoof)")

if __name__ == "__main__":
    main()
