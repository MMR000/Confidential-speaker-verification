# tools/pad_aasist_infer.py  (for speechbrain 0.5.x)
import argparse, os, sys, pandas as pd, torch, torchaudio
from tqdm import tqdm

try:
    from speechbrain.pretrained import AASIST  # 0.5.x 提供
except Exception as e:
    print("[ERR] 当前 SpeechBrain 版本不包含 pretrained.AASIST，请使用方案 A 的版本 (0.5.x)，或改用方案 B 的脚本。", file=sys.stderr)
    raise

def _read_trials(trials_path: str):
    df = pd.read_csv(trials_path, sep="\t")
    # 兼容两种列名：label/path 或 第一列为label、第二列为path
    path_col = "path" if "path" in df.columns else df.columns[1]
    label_col = "label" if "label" in df.columns else df.columns[0]
    if df[label_col].dtype == object:
        y = df[label_col].astype(str).str.lower().map(lambda s: 1 if ("real" in s or "bona" in s) else 0).astype(int)
    else:
        y = df[label_col].astype(int)
    paths = df[path_col].astype(str)
    return paths.tolist(), y.tolist()

def _to_mono16k(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
    return wav

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[LOAD] SpeechBrain AASIST (0.5.x) -> device={args.device}")
    model = AASIST.from_hparams(
        source="speechbrain/aasist-antispoofing",
        savedir=os.path.join(os.path.dirname(args.out), "aasist_ckpt"),
        run_opts={"device": args.device},
    )

    paths, labels = _read_trials(args.trials)
    rows = []
    for p, y in tqdm(list(zip(paths, labels)), total=len(paths), desc="AASIST infer"):
        try:
            wav, sr = torchaudio.load(p)
            wav = _to_mono16k(wav, sr).to(torch.float32).to(args.device)
            with torch.no_grad():
                probs, logits = model.classify_batch(wav.unsqueeze(0))  # 0.5.x 返回 (probs, logits)
                spoof_prob = float(probs[0, 1].detach().cpu())         # 索引1=spoof 概率
            rows.append([spoof_prob, y, p])
        except Exception as e:
            # 有坏文件就跳过该条
            # 如需日志：print(f"[WARN] {p}: {e}", file=sys.stderr)
            continue

    df_out = pd.DataFrame(rows, columns=["score", "label(optional)", "path"])
    df_out.to_csv(args.out, sep="\t", index=False)
    print(f"[OK] 写出：{args.out}  行数={len(df_out)}\n[NOTE] score 越大=越“假/合成”(spoof)")

if __name__ == "__main__":
    main()
