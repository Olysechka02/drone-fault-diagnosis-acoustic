"""
Инференс: классификация одного аудиофайла обученной моделью FusionResNetSEMTLCNN.

Pipeline:
    WAV → extract_mel_features() → (n_windows, 1280)
        → extract_custom_features() → (n_windows, 32)
        → StandardScaler (mel и custom) — обученные на train-выборке
        → reshape мел: (n_windows, 1280) → (n_windows, 20, 64)
        → FusionResNetSEMTLCNN → softmax → argmax
        → fault_encoder/maneuver_encoder.inverse_transform

Файл может содержать несколько окон (для 0.5 с при WIN_LEN=20, HOP_LEN=12 —
обычно 3 окна). Возвращается агрегированная оценка по всем окнам
(усреднение вероятностей).

Usage:
    python -m src.predict path/to/audio.wav

Зависимости (из weights/):
    fusion_resnet_se_mtl_cnn_weights.pth   — state_dict модели
    fault_encoder.pkl                       — LabelEncoder для 9 классов
    maneuver_encoder.pkl                    — LabelEncoder для 6 манёвров
    scaler_mel.pkl                          — StandardScaler для 1280-мерных мел-векторов
    scaler_custom.pkl                       — StandardScaler для 32-мерных STFT-векторов
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import torch

from .model import FusionResNetSEMTLCNN
from .features import extract_all, WIN_LEN, N_MELS

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def load_pipeline(weights_dir: Path = WEIGHTS_DIR, device: str = "cpu"):
    """Загрузка обученной модели, энкодеров классов и скейлеров признаков.

    Все .pkl-файлы сохранены через joblib (так делает sklearn по умолчанию).

    Returns:
        (model, fault_encoder, maneuver_encoder, scaler_mel, scaler_custom)
    """
    fault_enc = joblib.load(weights_dir / "fault_encoder.pkl")
    maneuver_enc = joblib.load(weights_dir / "maneuver_encoder.pkl")
    scaler_mel = joblib.load(weights_dir / "scaler_mel.pkl")
    scaler_custom = joblib.load(weights_dir / "scaler_custom.pkl")

    model = FusionResNetSEMTLCNN(
        n_fault_classes=len(fault_enc.classes_),
        n_maneuver_classes=len(maneuver_enc.classes_),
    )
    state = torch.load(
        weights_dir / "fusion_resnet_se_mtl_cnn_weights.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state)
    model.eval().to(device)

    return model, fault_enc, maneuver_enc, scaler_mel, scaler_custom


def predict(audio_path: str, device: str = "cpu") -> dict:
    """Классификация одного WAV-файла.

    Args:
        audio_path: путь к .wav (моно/стерео, любая частота — будет приведено к 16 кГц).
        device: 'cpu' или 'cuda'.
    Returns:
        Словарь:
            fault_class, fault_confidence, fault_probs (по всем классам)
            maneuver_class, maneuver_confidence, maneuver_probs
            n_windows (число окон, по которым усреднено)
    """
    model, fault_enc, maneuver_enc, scaler_mel, scaler_custom = load_pipeline(device=device)

    # 1. Извлечение признаков из WAV (точно как в обучающем pipeline)
    mel_flat, custom = extract_all(audio_path)        # (n_windows, 1280), (n_windows, 32)
    n_windows = len(mel_flat)

    # 2. Применяем сохранённые скейлеры (обучены на train-выборке)
    mel_scaled = scaler_mel.transform(mel_flat).astype(np.float32)
    custom_scaled = scaler_custom.transform(custom).astype(np.float32)

    # 3. Reshape мел: (n, 1280) → (n, 20, 64)
    mel_3d = mel_scaled.reshape(-1, WIN_LEN, N_MELS)

    # 4. Передача через модель
    with torch.no_grad():
        mel_t = torch.tensor(mel_3d, dtype=torch.float32).to(device)
        cust_t = torch.tensor(custom_scaled, dtype=torch.float32).to(device)
        fault_logits, maneuver_logits = model(mel_t, cust_t)

        fault_probs_each = torch.softmax(fault_logits, dim=1).cpu().numpy()
        maneuver_probs_each = torch.softmax(maneuver_logits, dim=1).cpu().numpy()

    # 5. Усредняем вероятности по окнам (soft voting)
    fault_probs = fault_probs_each.mean(axis=0)
    maneuver_probs = maneuver_probs_each.mean(axis=0)

    fault_idx = int(fault_probs.argmax())
    maneuver_idx = int(maneuver_probs.argmax())

    return {
        "fault_class": str(fault_enc.classes_[fault_idx]),
        "fault_confidence": float(fault_probs[fault_idx]),
        "fault_probs": {str(c): float(p) for c, p in zip(fault_enc.classes_, fault_probs)},
        "maneuver_class": str(maneuver_enc.classes_[maneuver_idx]),
        "maneuver_confidence": float(maneuver_probs[maneuver_idx]),
        "maneuver_probs": {str(c): float(p) for c, p in zip(maneuver_enc.classes_, maneuver_probs)},
        "n_windows": int(n_windows),
    }


if __name__ == "__main__":
    # 📌 По умолчанию пример — данные/samples/. Замените на свой путь к WAV.
    if len(sys.argv) < 2:
        default = Path(__file__).resolve().parent.parent / "data" / "samples" / \
                  "A_B_MF1_185_DuckPond_637_snr=13.182254877166207.wav"
        if default.exists():
            print(f"Путь не задан. Использую пример: {default.name}")
            audio_path = str(default)
        else:
            print("Usage: python -m src.predict <path_to_audio.wav>")
            sys.exit(1)
    else:
        audio_path = sys.argv[1]

    result = predict(audio_path)
    print(f"\nФайл: {Path(audio_path).name}")
    print(f"Обработано окон: {result['n_windows']}")
    print()
    print(f"  Неисправность: {result['fault_class']} "
          f"(confidence {result['fault_confidence']:.4f})")
    print(f"  Манёвр:        {result['maneuver_class']} "
          f"(confidence {result['maneuver_confidence']:.4f})")

    print("\n  Top-3 fault probabilities:")
    top3 = sorted(result["fault_probs"].items(), key=lambda kv: -kv[1])[:3]
    for cls, p in top3:
        print(f"    {cls}: {p:.4f}")
