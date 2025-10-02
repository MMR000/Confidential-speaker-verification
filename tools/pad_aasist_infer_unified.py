#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified PAD inference with AASIST-family (robust, offline-friendly).

优先级：
1) --repo_dir + --ckpt  -> 本地 AASIST3 源码 + safetensors
2) --sb_dir             -> 本地 SpeechBrain AASIST 快照目录（含 hyperparams.yaml / *.py / ckpt）
3) 线上拉取             -> 依次尝试多个 HF 仓库名（需要可用网络 + 可访问）

输出：TSV，三列固定：score  label(optional)  path
分数定义：数值越大 => 越可能是 spoof
"""

import os, sys, re, argparse, importlib.util, glob, json
import torch, torchaudio
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Util
# -----------------------------
def read_trials(trials_path: str):
    """尽量鲁棒地读 TSV，推断 path & label 列；label: 1=bonafide/real, 0=spoof/synth/tts"""
    df = pd.read_csv(trials_path, sep="\t")
    # 找 path 列：优先 'path'，其次包含斜杠的那列
    if "path" in df.columns:
        path_col = "path"
    else:
        path_col = None
        for c in df.columns:
            if df[c].astype(str).str.contains(r"[\\/]").mean() > 0.5:
                path_col = c
                break
        if path_col is None:
            # 退而求其次：第一列
            path_col = df.columns[0]
    # 找 label 列：优先 'label'，否则挑一个非 path 列
    label_col = "label" if "label" in df.columns else None
    if label_col is None:
        for c in df.columns:
            if c != path_col:
                label_col = c
                break
    paths = df[path_col].astype(str).tolist()
    lab = df[label_col]
    if lab.dtype == object:
        y = lab.astype(str).str.lower().map(lambda s: 1 if ("real" in s or "bona" in s) else 0).fillna(0).astype(int)
    else:
        y = lab.astype(int)
    return paths, y.tolist()

def to_mono_16k_chunk(audio: torch.Tensor, sr: int, target_len: int = 64600) -> torch.Tensor:
    if audio.dim() == 2:
        audio = audio.mean(dim=0)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio.unsqueeze(0), sr, 16000).squeeze(0)
    T = audio.shape[-1]
    if T < target_len:
        audio = torch.nn.functional.pad(audio, (0, target_len - T))
    else:
        audio = audio[:target_len]
    return audio

def _dynamic_import(py_path: str, module_name: str = None):
    if module_name is None:
        module_name = os.path.splitext(os.path.basename(py_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# -----------------------------
# Backend A：本地 AASIST3 源码 + safetensors
# -----------------------------
def load_local_aasist3(repo_dir: str, ckpt_path: str, device: str):
    if not os.path.isdir(repo_dir):
        raise RuntimeError(f"repo_dir not found: {repo_dir}")
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"ckpt not found: {ckpt_path}")

    py_candidates = []
    for pat in ["model/*.py", "models/*.py", "*.py"]:
        py_candidates.extend(glob.glob(os.path.join(repo_dir, pat)))

    target_cls = None
    target_mod = None
    for py in py_candidates:
        try:
            txt = open(py, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        if re.search(r"class\s+(aasist3|AASIST3)\(", txt):
            mod = _dynamic_import(py)
            if hasattr(mod, "aasist3"):
                target_cls = getattr(mod, "aasist3")
                target_mod = py
                break
            if hasattr(mod, "AASIST3"):
                target_cls = getattr(mod, "AASIST3")
                target_mod = py
                break
    if target_cls is None:
        raise RuntimeError("未在 repo_dir 中找到 class aasist3/AASIST3（请确认源码结构）。")

    # 构建模型
    if hasattr(target_cls, "from_pretrained"):
        model = target_cls.from_pretrained(repo_dir)
    else:
        cfg = {}
        cfg_path = os.path.join(repo_dir, "config.json")
        if os.path.isfile(cfg_path):
            try:
                cfg = json.load(open(cfg_path, "r"))
            except Exception:
                cfg = {}
        model = target_cls(**cfg) if cfg else target_cls()

    # 加载权重
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(ckpt_path)
    else:
        state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    print(f"[LOAD] Local AASIST3 OK from {target_mod} + {ckpt_path}")
    return ("aasist3_local", model)

def aasist3_infer(model, wavs: torch.Tensor):
    out = model(wavs)  # 期望 [B, 2] logits
    if isinstance(out, (list, tuple)):
        out = out[0]
    if out.dim() == 2 and out.shape[-1] == 2:
        probs = torch.softmax(out, dim=-1)[:, 1]
    elif out.dim() == 1:
        probs = out
    else:
        raise RuntimeError(f"Unexpected AASIST3 output shape {tuple(out.shape)}")
    return probs

# -----------------------------
# Backend B：本地 SpeechBrain AASIST 快照目录
# -----------------------------
def load_sb_local(sb_dir: str, device: str):
    if not os.path.isdir(sb_dir):
        raise RuntimeError(f"sb_dir not found: {sb_dir}")
    # 在本地快照里找一个定义 AASIST 的接口（class AASIST 带 from_hparams）
    py_candidates = []
    for pat in ["*.py", "interface/*.py", "inference*.py", "custom*.py", "recipes/*/*.py"]:
        py_candidates.extend(glob.glob(os.path.join(sb_dir, pat)))

    cls = None
    mod_found = None
    for py in py_candidates:
        try:
            txt = open(py, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        if re.search(r"class\s+AASIST\(", txt):
            mod = _dynamic_import(py)
            if hasattr(mod, "AASIST") and hasattr(getattr(mod, "AASIST"), "from_hparams"):
                cls = getattr(mod, "AASIST")
                mod_found = py
                break
    if cls is None:
        # 再宽松点：找有 from_hparams 的类
        for py in py_candidates:
            try:
                mod = _dynamic_import(py, module_name="sb_iface_" + os.path.basename(py).replace(".py",""))
            except Exception:
                continue
            for name in dir(mod):
                obj = getattr(mod, name)
                if isinstance(obj, type) and hasattr(obj, "from_hparams"):
                    cls = obj
                    mod_found = py
                    break
            if cls is not None:
                break
    if cls is None:
        raise RuntimeError("未在 sb_dir 中找到 SpeechBrain AASIST 接口（含 from_hparams）。")

    try:
        model = cls.from_hparams(source=sb_dir, savedir=sb_dir, run_opts={"device": device})
        model.device = device
        print(f"[LOAD] SpeechBrain AASIST (local) OK from {sb_dir} via {mod_found}")
        return ("speechbrain", model)
    except Exception as e:
        raise RuntimeError(f"SpeechBrain AASIST 本地加载失败: {repr(e)}")

def sb_infer(model, wavs: torch.Tensor):
    # 找推理接口
    out = None
    for fn in ["classify_batch", "forward", "predict", "detect"]:
        if hasattr(model, fn):
            out = getattr(model, fn)(wavs)
            break
    if out is None:
        raise RuntimeError("SpeechBrain 接口未找到 classify/predict 方法。")

    # 统一成 spoof 概率
    if isinstance(out, dict):
        if "probs" in out:
            probs = out["probs"]
        elif "scores" in out:
            probs = torch.softmax(out["scores"], dim=-1)
        else:
            probs = None
            for v in out.values():
                if torch.is_tensor(v):
                    probs = v
                    break
            if probs is None:
                raise RuntimeError("无法解析 SpeechBrain 输出字典。")
    else:
        probs = out

    if probs.dim() == 2 and probs.shape[-1] == 2:
        spoof_prob = torch.softmax(probs, dim=-1)[:, 1]
    elif probs.dim() == 1:
        spoof_prob = probs
    else:
        raise RuntimeError(f"Unexpected SB output shape: {tuple(probs.shape)}")
    return spoof_prob

# -----------------------------
# Backend C：在线拉取多个候选 HF 仓库
# -----------------------------
HF_SB_CANDIDATES = [
    # 逐个尝试（不同版本/命名）
    "speechbrain/antispoofing-AASIST",
    "speechbrain/antispoofing-AASIST-LA",
    "speechbrain/AntiSpoofing-AASIST",
    "speechbrain/AASIST-ASVspoof2019-LA",
]

def load_sb_online(device: str, repo_id: str = None):
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        raise RuntimeError("缺少 huggingface_hub，请先 pip install huggingface_hub")
    errs = []
    if repo_id:
        candidates = [repo_id]
    else:
        candidates = HF_SB_CANDIDATES
    for rid in candidates:
        try:
            local_dir = snapshot_download(repo_id=rid, local_files_only=False)
            print(f"[HF] 下载完成：{rid} -> {local_dir}")
            return load_sb_local(local_dir, device)
        except Exception as e:
            errs.append(f"{rid}: {e}")
    raise RuntimeError("在线拉取 SpeechBrain 失败：\n" + "\n".join(errs))

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True, help="TSV with path & label")
    ap.add_argument("--out", required=True, help="Output TSV (score\tlabel(optional)\tpath)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--win_len", type=int, default=64600)

    # AASIST3 本地
    ap.add_argument("--repo_dir", default=None, help="Local AASIST3 source dir")
    ap.add_argument("--ckpt", default=None, help="AASIST3 checkpoint (.safetensors/.pth)")

    # SpeechBrain 本地
    ap.add_argument("--sb_dir", default=None, help="Local snapshot dir of SpeechBrain AASIST")

    # SpeechBrain 在线（可选）
    ap.add_argument("--hf_repo", default=None, help="Override HF repo_id for SpeechBrain AASIST")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 依次尝试三条路径
    model = None
    backend = None
    # 1) 本地 AASIST3
    if args.repo_dir and args.ckpt:
        try:
            backend, model = load_local_aasist3(args.repo_dir, args.ckpt, args.device)
        except Exception as e:
            print(f"[WARN] 本地 AASIST3 失败：{e}")

    # 2) 本地 SpeechBrain
    if model is None and args.sb_dir:
        try:
            backend, model = load_sb_local(args.sb_dir, args.device)
        except Exception as e:
            print(f"[WARN] 本地 SpeechBrain 失败：{e}")

    # 3) 在线拉取 SpeechBrain
    if model is None:
        try:
            backend, model = load_sb_online(args.device, repo_id=args.hf_repo)
        except Exception as e:
            print(f"[ERR] 在线 SpeechBrain 失败：{e}")
            print(">> 提示：可以先离线下载到本地，再用 --sb_dir 指定本地目录：")
            print("   huggingface-cli download speechbrain/antispoofing-AASIST --local-dir sb_aasist")
            print("   （或将上面命令的仓库名换成其它候选，再把 --sb_dir=sb_aasist 传给本脚本）")
            sys.exit(1)

    # 读取 trials
    paths, labels = read_trials(args.trials)

    # 推理
    rows = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), args.batch), desc=f"{backend} infer"):
            batch_paths = paths[i : i + args.batch]
            batch_labels = labels[i : i + args.batch]
            wavs = []
            keep = []
            for k, p in enumerate(batch_paths):
                try:
                    audio, sr = torchaudio.load(p)
                    audio = to_mono_16k_chunk(audio, sr, args.win_len)
                    wavs.append(audio)
                    keep.append(k)
                except Exception:
                    continue
            if not keep:
                continue
            x = torch.stack(wavs, dim=0).to(torch.float32).to(args.device)
            if backend == "speechbrain":
                probs = sb_infer(model, x)
            else:
                probs = aasist3_infer(model, x)
            probs = probs.detach().cpu().tolist()
            for j, idx in enumerate(keep):
                rows.append([probs[j], batch_labels[idx], batch_paths[idx]])

    df = pd.DataFrame(rows, columns=["score", "label(optional)", "path"])
    df.to_csv(args.out, sep="\t", index=False)
    print(f"[OK] Wrote: {args.out}  rows={len(df)}")
    print("[NOTE] Higher score => more spoof-like")

if __name__ == "__main__":
    main()
