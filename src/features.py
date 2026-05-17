"""
Извлечение акустических признаков для FusionResNetSEMTLCNN.

ВНИМАНИЕ: пайплайны 1-в-1 воспроизводят оригинальные скрипты, которыми были
получены признаки для обучения. Иначе StandardScaler не подойдёт к новым данным.

Два пайплайна:
1. extract_mel_features() — мел-спектрограмма 20 кадров × 64 полосы = 1280 значений
   (через torchaudio.transforms.MelSpectrogram + AmplitudeToDB).
2. extract_custom_features() — 32 статистических STFT-признака
   (через librosa.stft + покадровые признаки + агрегация mean/std/max/min).

Параметры по умолчанию соответствуют тренировочному пайплайну:
SR = 16 кГц, N_FFT = 512, HOP_LENGTH = 160 (10 мс), WIN_LEN = 20 кадров.
"""
from typing import Optional
import numpy as np
import torch
import torchaudio.transforms as T
import soundfile as sf
import librosa
from scipy.stats import gmean

# ── Параметры (1-в-1 как в обучающих скриптах) ─────────────────────────────
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160          # 10 мс при 16 кГц
N_MELS = 64
WIN_LEN = 20              # 20 кадров = 0,20 с
HOP_LEN = 12              # 12 кадров между окнами (40% перекрытие)

# Узкополосный диапазон для custom-признаков (2–3 кГц)
BIN_MIN = int(2000 / (SAMPLE_RATE / N_FFT))   # ~64
BIN_MAX = int(3000 / (SAMPLE_RATE / N_FFT))   # ~96

# torchaudio-преобразование (один раз, без повторного создания)
_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)
_to_db = T.AmplitudeToDB()


def load_audio_to_mono(audio_path: str) -> tuple[np.ndarray, int]:
    """Загрузка WAV → моно-сигнал 16 кГц (numpy, float)."""
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != SAMPLE_RATE:
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
    return data.astype(np.float32), SAMPLE_RATE


def extract_mel_features(audio_path: str) -> np.ndarray:
    """Извлечь мел-признаки в формате обучающего pipeline.

    Возвращает массив окон (n_windows, 1280) — flat-вид, как при обучении.
    Каждое окно — 20 кадров × 64 мел-полосы.

    Если файл короче WIN_LEN кадров (0,2 с), он будет дополнен нулями.
    """
    data, _ = load_audio_to_mono(audio_path)

    waveform = torch.from_numpy(data).float().unsqueeze(0)  # (1, N)

    spec = _mel_transform(waveform)               # (1, n_mels, time)
    spec_db = _to_db(spec)                         # (1, n_mels, time)
    # (n_mels, time) → (time, n_mels)
    features = spec_db.squeeze(0).T.numpy()        # (time, n_mels)

    # Разбиение на окна WIN_LEN × N_MELS с шагом HOP_LEN
    if features.shape[0] < WIN_LEN:
        pad = WIN_LEN - features.shape[0]
        features = np.pad(features, ((0, pad), (0, 0)), mode="constant")

    N = features.shape[0]
    windows = []
    start = 0
    while start + WIN_LEN <= N:
        window = features[start:start + WIN_LEN].reshape(-1)  # → 1280
        windows.append(window)
        start += HOP_LEN

    if not windows:
        windows = [features[:WIN_LEN].reshape(-1)]

    return np.array(windows, dtype=np.float32)     # (n_windows, 1280)


def _extract_frame_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Покадровые 8 признаков для одного аудиосигнала.

    Возвращает массив (n_frames, 8): band_energy, band_flatness, band_peak_freq,
    zcr, centroid, bandwidth, rolloff, rms.
    """
    stft_matrix = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    power_spec = np.abs(stft_matrix) ** 2

    # Узкополосные (2–3 кГц)
    band_spec = power_spec[BIN_MIN:BIN_MAX, :]
    band_energy = np.sum(band_spec, axis=0)

    amean = np.mean(band_spec + 1e-10, axis=0)
    g_mean = gmean(band_spec + 1e-10, axis=0)
    band_flatness = g_mean / amean

    peak_bins = np.argmax(band_spec, axis=0) + BIN_MIN
    band_peak_freq = peak_bins * (sr / N_FFT)

    # Широкополосные
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(S=np.abs(stft_matrix), sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=np.abs(stft_matrix), sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=np.abs(stft_matrix), sr=sr, roll_percent=0.85)[0]
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]

    min_len = min(len(band_energy), len(zcr), len(centroid),
                  len(bandwidth), len(rolloff), len(rms))

    return np.vstack([
        band_energy[:min_len],
        band_flatness[:min_len],
        band_peak_freq[:min_len],
        zcr[:min_len],
        centroid[:min_len],
        bandwidth[:min_len],
        rolloff[:min_len],
        rms[:min_len],
    ]).T  # (n_frames, 8)


def extract_custom_features(audio_path: str) -> np.ndarray:
    """Извлечь 32 STFT-статистических признака.

    Возвращает массив окон (n_windows, 32): по 8 базовых признаков × 4 статистики
    (mean, std, max, min) для каждого окна WIN_LEN кадров.
    """
    data, _ = load_audio_to_mono(audio_path)

    frame_feats = _extract_frame_features(data, SAMPLE_RATE)

    if frame_feats.shape[0] < WIN_LEN:
        pad = WIN_LEN - frame_feats.shape[0]
        frame_feats = np.pad(frame_feats, ((0, pad), (0, 0)), mode="constant")

    N = frame_feats.shape[0]
    windows = []
    start = 0
    while start + WIN_LEN <= N:
        window = frame_feats[start:start + WIN_LEN]
        w_mean = np.mean(window, axis=0)
        w_std = np.std(window, axis=0)
        w_max = np.max(window, axis=0)
        w_min = np.min(window, axis=0)
        windows.append(np.concatenate([w_mean, w_std, w_max, w_min]))
        start += HOP_LEN

    if not windows:
        windows.append(np.concatenate([
            np.mean(frame_feats, axis=0),
            np.std(frame_feats, axis=0),
            np.max(frame_feats, axis=0),
            np.min(frame_feats, axis=0),
        ]))

    return np.array(windows, dtype=np.float32)     # (n_windows, 32)


def extract_all(audio_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Полный pipeline: путь к WAV → (мел-окна 1280, custom-окна 32).

    Возвращает два numpy-массива одинаковой длины по первому измерению
    (n_windows, 1280) и (n_windows, 32).
    """
    mel = extract_mel_features(audio_path)
    custom = extract_custom_features(audio_path)
    n = min(len(mel), len(custom))
    return mel[:n], custom[:n]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.features <path_to_wav>")
        sys.exit(1)
    mel, custom = extract_all(sys.argv[1])
    print(f"Мел-признаки: {mel.shape}   диапазон [{mel.min():.1f}, {mel.max():.1f}] дБ")
    print(f"Custom-признаки: {custom.shape}")
    print(f"Первые 5 custom: {custom[0][:5]}")
