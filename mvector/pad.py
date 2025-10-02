# mvector/pad.py
# 依赖：pip install -U numpy soundfile librosa

import numpy as np
import librosa
import soundfile as sf  # 只为确保依赖完整

def _safe_read(wav_path: str):
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    if not np.isfinite(y).all():
        y = np.nan_to_num(y)
    if y.size == 0:
        y = np.zeros(16000, dtype=np.float32)
    y = librosa.util.normalize(y.astype(np.float32))
    return y, 16000

def _feat_score(y: np.ndarray, sr: int) -> float:
    # 简单启发式：过零率 + 频谱平坦度 + VAD 覆盖（越大越“真人/bonafide”）
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=400, hop_length=160)))
    zcr_term = 1.0 - min(abs(zcr - 0.08) / 0.08, 1.0)  # 0~1

    S = np.abs(librosa.stft(y, n_fft=512, hop_length=160)) + 1e-12
    flatness = np.exp(np.mean(np.log(S), axis=0)) / (np.mean(S, axis=0) + 1e-12)
    flat = float(np.clip(1.0 - np.mean(flatness), 0.0, 1.0))  # 越大越“有结构”

    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    vad = float(np.mean(rms > (np.median(rms) + 0.5*np.std(rms) + 1e-6)))  # 0~1

    raw = 0.5 * zcr_term + 0.35 * flat + 0.15 * vad
    score = 1.0 / (1.0 + np.exp(-8.0 * (raw - 0.5)))
    return float(np.clip(score, 0.0, 1.0))

class PADPredictor:
    def __init__(self):
        pass  # 无外部权重，开箱即用

    def predict(self, wav_path: str) -> float:
        """返回 bonafide 概率（[0,1]），越大越像真人"""
        y, sr = _safe_read(wav_path)
        return _feat_score(y, sr)
