# tools/build_pad_trials.py
# 用法示例：
#   python tools/build_pad_trials.py \
#     --root ~/Desktop/VoiceClass/train --split train \
#     --out output/pad_trials_train.tsv
#   python tools/build_pad_trials.py \
#     --root ~/Desktop/VoiceClass/val --split val \
#     --out output/pad_trials_val.tsv

import os
import argparse

AUDIO_EXTS = (".wav",".flac",".mp3",".m4a",".ogg",".WAV",".FLAC",".MP3",".M4A",".OGG")

def collect_paths(d):
    out = []
    for r,_,fs in os.walk(d):
        for f in fs:
            if f.endswith(AUDIO_EXTS):
                out.append(os.path.join(r,f))
    return sorted(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="包含 real / cloned / Synthesized 的目录（如 .../VoiceClass/train）")
    ap.add_argument("--split", default="train", help="仅用于文件名标识")
    ap.add_argument("--out", default=None, help="输出 TSV 路径；缺省则写 output/pad_trials_<split>.tsv")
    ap.add_argument("--label_style", choices=["word","01"], default="word",
                    help="标签风格：word=bonafide/spoof，01=1/0")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    real_dir = os.path.join(root, "real")
    #cloned_dir = os.path.join(root, "cloned")
    synth_dir = os.path.join(root, "Synthesized")

    if not os.path.isdir(root):
        raise SystemExit(f"[ERR] root 不存在：{root}")

    out_path = args.out or os.path.join("output", f"pad_trials_{args.split}.tsv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def lab_bona(): return "bonafide" if args.label_style=="word" else "1"
    def lab_spoof(): return "spoof" if args.label_style=="word" else "0"

    pairs = []
    if os.path.isdir(real_dir):
        for p in collect_paths(real_dir):
            pairs.append((p, lab_bona()))
    # if os.path.isdir(cloned_dir):
    #     for p in collect_paths(cloned_dir):
    #         pairs.append((p, lab_spoof()))
    if os.path.isdir(synth_dir):
        for p in collect_paths(synth_dir):
            pairs.append((p, lab_spoof()))

    if not pairs:
        raise SystemExit(f"[ERR] 在 {root} 下未找到任何音频。期望子目录：real/、cloned/、Synthesized/")

    with open(out_path, "w", encoding="utf-8") as w:
        for p,l in pairs:
            w.write(f"{p}\t{l}\n")

    print(f"[OK] 写出：{out_path}  共 {len(pairs)} 行")
    if os.path.isdir(real_dir):      print(f"  real:        {len(collect_paths(real_dir))}")
    # if os.path.isdir(cloned_dir):    print(f"  cloned:      {len(collect_paths(cloned_dir))}")
    if os.path.isdir(synth_dir):     print(f"  Synthesized: {len(collect_paths(synth_dir))}")

if __name__ == "__main__":
    main()
