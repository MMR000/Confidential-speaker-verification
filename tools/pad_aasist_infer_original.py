# tools/pad_aasist_infer_original.py
import os, sys, argparse, json, math
import torch
import torch.nn.functional as F
import torchaudio

def load_model(aasist_dir: str, device: str = "cuda"):
    """从 clovaai/aasist 仓库加载原始 AASIST 模型与预训练权重。"""
    aasist_dir = os.path.abspath(aasist_dir)
    assert os.path.isdir(aasist_dir), f"aasist_dir not found: {aasist_dir}"
    sys.path.insert(0, aasist_dir)

    # 导入模型定义
    from models.AASIST import Model

    # 按仓库默认配置（关键超参）
    d_args = {
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "n_classes": 2,
        "nb_samp": 64600,   # 4秒@16kHz
    }
    print("[INFO] 使用的关键参数：", d_args)

    model = Model(d_args).to(device)
    model.eval()

    # 预训练权重（仓库自带）
    ckpt = os.path.join(aasist_dir, "models", "weights", "AASIST.pth")
    if os.path.isfile(ckpt):
        pkg = torch.load(ckpt, map_location="cpu")
        # 兼容 state_dict 包装
        state = pkg.get("model", pkg)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] load_state_dict: missing={missing}, unexpected={unexpected}")
        print(f"[OK] 已加载预训练权重：{ckpt}")
    else:
        print(f"[WARN] 未找到权重 {ckpt}，将用随机初始化（指标会差很多）")

    return model, d_args

def load_wave(path: str, target_sr=16000, nb_samp=64600, device="cuda"):
    """读取原始波形 -> 单声道 -> 16k -> 裁剪/补齐到 nb_samp -> [1, 1, nb_samp]"""
    wav, sr = torchaudio.load(path)  # [C, T]
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # -> [1, T]

    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)

    # 归一化到 [-1,1]（可选）
    maxv = wav.abs().max()
    if maxv > 0:
        wav = wav / maxv

    T = wav.shape[1]
    if T < nb_samp:
        wav = F.pad(wav, (0, nb_samp - T))
    else:
        wav = wav[:, :nb_samp]

    # 模型期望 [B, 1, nb_samp]
    wav = wav.unsqueeze(0)  # [1, 1, nb_samp]
    return wav.to(device)

def read_paths(trials_path: str):
    """trials 文件每行一个 .wav 路径（或混合行，自动抽出最后一个 .wav）"""
    import re
    paths=[]
    with open(trials_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            toks = re.split(r"[\t ,]+", line)
            wavs = [t for t in toks if t.lower().endswith(".wav")]
            if wavs:
                paths.append(wavs[-1])
            elif os.path.isfile(line):
                paths.append(line)
            # 否则跳过
    if not paths:
        raise RuntimeError(f"没有在 {trials_path} 里找到任何 .wav 路径")
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True, help="每行一个 wav 路径（或包含 wav 的行）")
    ap.add_argument("--out", required=True, help="输出 TSV：score\\tpath")
    ap.add_argument("--aasist_dir", required=True, help="clovaai/aasist 仓库路径")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model, d_args = load_model(args.aasist_dir, device=device)
    nb_samp = d_args["nb_samp"]

    paths = read_paths(args.trials)
    print(f"[INFO] 将对 {len(paths)} 个文件做推理；示例：")
    for p in paths[:5]:
        print("   ", p)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")

    model.eval()
    with torch.no_grad():
        buf = []
        meta = []
        for i, p in enumerate(paths, 1):
            try:
                x = load_wave(p, target_sr=16000, nb_samp=nb_samp, device=device)  # [1,1,nb_samp]
            except Exception as e:
                print(f"[ERR] 读文件失败：{p} -> {e}")
                continue

            buf.append(x)
            meta.append(p)

            if len(buf) == args.batch or i == len(paths):
                X = torch.cat(buf, dim=0)  # [B,1,nb_samp]
                # 一次性 forward
                logits = model(X)          # [B,2]
                if isinstance(logits, dict):
                    logits = logits.get("logits", None)
                    assert logits is not None, "model 返回字典但没有 'logits' 键"
                probs = torch.softmax(logits, dim=1)
                # 约定：index 1 = spoof 概率；如果你要 bonafide 概率就用 probs[:,0]
                scores = probs[:, 1].detach().float().cpu().tolist()

                for s, path in zip(scores, meta):
                    fout.write(f"{s:.6f}\t{path}\n")

                buf.clear(); meta.clear()

    fout.close()
    print(f"[OK] 已写出：{args.out}")

if __name__ == "__main__":
    main()
