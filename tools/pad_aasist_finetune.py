# tools/pad_aasist_finetune.py
import argparse, os, torch, torchaudio, pandas as pd, numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def read_tsv(tsv):
    df = pd.read_csv(tsv, sep="\t")
    path_col = "path" if "path" in df.columns else df.columns[1]
    label_col = "label" if "label" in df.columns else df.columns[0]
    if df[label_col].dtype == object:
        y = df[label_col].astype(str).str.lower().map(lambda s: 1 if "real" in s or "bona" in s else 0).values
    else:
        y = df[label_col].astype(int).values
    x = df[path_col].astype(str).values
    return x, y

class PADSet(Dataset):
    def __init__(self, tsv):
        self.paths, self.labels = read_tsv(tsv)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]; y = int(self.labels[i])
        wav, sr = torchaudio.load(p)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.squeeze(0), y

class Head(nn.Module):
    def __init__(self, in_dim=256, hidden=128):  # AASIST embedding 维度会在运行时探测
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2),
        )
    def forward(self, x): return self.net(x)

@torch.no_grad()
def embed_batch(aasist, waves, device):
    # 从 AASIST 抽 embedding：用内部的前馈把时域转为定长向量
    # SpeechBrain AASIST 没有官方公开 "encode" 接口，这里走 classify 前的 pooled 表征：
    # 兼容：拿 logits 前一层的 pooled 特征（model.mods.* 组合），最稳方式是：取 classify_batch 返回的 logits 前的 penultimate。
    # 但 speechbrain API 未直接暴露；折中：我们用 logits 之前的线性层输入作为 embedding。
    # 为简化演示，我们直接用 model.mods.classifier[-1].weight.shape[1] 作为维度，并通过 hook 抓取倒数第二层输入。
    # —— 若版本差异导致失败，改“直接用 logits 作为特征”（效果略差，但能训）。
    out_prob, out_logits = aasist.classify_batch(waves)  # logits: [B,2]
    return out_logits  # 当作 2 维特征（简单但鲁棒）。若你要更强，可加 hook 抓 penultimate。

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--val_tsv", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from speechbrain.pretrained import AASIST
    os.makedirs(args.save_dir, exist_ok=True)
    aasist = AASIST.from_hparams(
        source="speechbrain/aasist-antispoofing",
        savedir=os.path.join(args.save_dir, "aasist_ckpt"),
        run_opts={"device": args.device}
    )
    # 冻结
    for p in aasist.modules.parameters(): p.requires_grad = False

    train_set = PADSet(args.train_tsv)
    val_set   = PADSet(args.val_tsv)
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=4)

    # 先探测特征维度（这里用 logits 作为特征，维度=2）
    feat_dim = 2
    head = Head(in_dim=feat_dim, hidden=128).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        head.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for waves, y in pbar:
            waves = torch.stack([w for w in waves], dim=0).to(args.device)
            y = y.to(args.device)

            with torch.no_grad():
                feats = embed_batch(aasist, waves, args.device)  # [B,2]
            logits = head(feats)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        # 验证
        head.eval(); n=0; correct=0
        with torch.no_grad():
            for waves, y in val_loader:
                waves = torch.stack([w for w in waves], dim=0).to(args.device)
                y = y.to(args.device)
                feats = embed_batch(aasist, waves, args.device)
                logits = head(feats)
                pred = logits.argmax(1)
                correct += (pred==y).sum().item()
                n += y.numel()
        acc = correct / max(1,n)
        print(f"[VAL] acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(head.state_dict(), os.path.join(args.save_dir, "best_head.pt"))
            print("[SAVE] best_head.pt")

    print(f"[DONE] best val acc = {best_acc:.4f}\n"
          f"用法：推理时先跑 AASIST logits，再过这个 head 得到二分类。")

if __name__ == "__main__":
    main()
