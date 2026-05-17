"""
Извлечение акустических признаков для FusionResNetSEMTLCNN.

Два независимых пайплайна:
1. extract_mel(...)  — мел-спектрограмма 20 кадров × 64 полосы = 1280 значений;
2. extract_stft_stats(...) — 32 статистических признака (3 узкополосных в полосе
   2–3 кГц + 5 широкополосных, агрегированных mean/std/max/min).

Параметры по умолчанию соответствуют исходному датасету Yi et al. (2023):
SR = 16 кГц, окно сегмента 0.5 с, HOP_LENGTH = 160 (10 мс), N_FFT = 512.
"""
import numpy as np
import librosa
import soundfile as sf

SR = 16000
N_FFT = 512
HOP_LEN = 160          # 10 мс при 16 кГц
N_MELS = 64
WIN_LEN_FRAMES = 20    # длина окна для агрегации (= 0.2 с)
SEGMENT_LEN_S = 0.5    # длина одного входного сегмента, с


def load_audio(path: str, sr: int = SR, duration: float = SEGMENT_LEN_S) -> np.ndarray:
    """Загрузка аудио, преобразование в моно с частотой sr, обрезка/паддинг до duration с."""
    y, native_sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if native_sr != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=native_sr, target_sr=sr)
    target_len = int(sr * duration)
    if len(y) >= target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    return y.astype(np.float32)


def extract_mel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Мел-спектрограмма → (frames, n_mels) = (20, 64).

    Args:
        y: моно-сигнал длиной 0.5 с.
        sr: частота дискретизации.
    Returns:
        Матрица (frames, n_mels) с log-mel значениями (dB).
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS, fmax=sr // 2
    )
    S_db = librosa.power_to_db(S, ref=np.max)  # (n_mels, n_frames)
    # Берём первые WIN_LEN_FRAMES кадров (20 кадров × 10 мс = 0.2 с)
    n_frames = S_db.shape[1]
    if n_frames >= WIN_LEN_FRAMES:
        S_db = S_db[:, :WIN_LEN_FRAMES]
    else:
        pad = WIN_LEN_FRAMES - n_frames
        S_db = np.pad(S_db, ((0, 0), (0, pad)), mode="edge")
    return S_db.T  # (WIN_LEN_FRAMES, N_MELS)


def extract_stft_stats(y: np.ndarray, sr: int = SR,
                       band_low: float = 2000.0, band_high: float = 3000.0) -> np.ndarray:
    """32 статистических STFT-признака.

    Состав:
    - 3 узкополосных признака в полосе band_low..band_high (band_energy, band_flatness, peak_freq);
    - 5 широкополосных: zcr, spectral_centroid, spectral_bandwidth, spectral_rolloff, rms;
    - каждый агрегируется четырьмя статистиками (mean, std, max, min) по окну WIN_LEN_FRAMES.

    Returns:
        Вектор длины 8 × 4 = 32.
    """
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LEN)
    mag = np.abs(D)                                                   # (n_freqs, n_frames)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # ── узкополосные: индексы частот band_low..band_high ────────────
    band_mask = (freqs >= band_low) & (freqs <= band_high)
    band_spec = mag[band_mask, :]                                     # (n_band_freqs, n_frames)

    band_energy = (band_spec ** 2).sum(axis=0)                        # (n_frames,)
    eps = 1e-10
    geom = np.exp(np.log(band_spec + eps).mean(axis=0))
    arith = band_spec.mean(axis=0) + eps
    band_flatness = geom / arith                                      # (n_frames,)
    band_peak = freqs[band_mask][band_spec.argmax(axis=0)]             # (n_frames,)

    # ── широкополосные ──────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LEN)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN, roll_percent=0.85)[0]
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LEN)[0]

    # ── агрегация ───────────────────────────────────────────────────
    base_features = [band_energy, band_flatness, band_peak, zcr, centroid, bandwidth, rolloff, rms]

    # выровняем длины (берём первые WIN_LEN_FRAMES)
    def _align(arr):
        if len(arr) >= WIN_LEN_FRAMES:
            return arr[:WIN_LEN_FRAMES]
        return np.pad(arr, (0, WIN_LEN_FRAMES - len(arr)), mode="edge")

    stats = []
    for feat in base_features:
        feat = _align(feat)
        stats.extend([feat.mean(), feat.std(), feat.max(), feat.min()])
    return np.asarray(stats, dtype=np.float32)  # (32,)


def extract_all(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Полный pipeline: путь к аудио → (мел_2D, stft_1D)."""
    y = load_audio(path)
    return extract_mel(y), extract_stft_stats(y)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m src.features <path_to_wav>")
        sys.exit(1)
    mel, stft = extract_all(sys.argv[1])
    print(f"Мел: {mel.shape}, диапазон [{mel.min():.1f}, {mel.max():.1f}] дБ")
    print(f"STFT-статистики: {stft.shape}, первые 5: {stft[:5]}")
