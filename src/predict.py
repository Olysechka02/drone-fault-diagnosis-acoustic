"""
Инференс: классификация одного аудиофайла обученной моделью FusionResNetSEMTLCNN.

Usage:
    python -m src.predict path/to/audio.wav

Output:
    Fault: PC1 (confidence 0.9953)
    Maneuver: B (confidence 0.9398)
"""
import sys
import pickle
from pathlib import Path
import numpy as np
import torch

from .model import FusionResNetSEMTLCNN
from .features import extract_all

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def load_pipeline(weights_dir: Path = WEIGHTS_DIR, device: str = "cpu"):
    """Загрузка обученной модели, энкодеров классов и скейлеров признаков."""
    model = FusionResNetSEMTLCNN()
    state = torch.load(weights_dir / "fusion_resnet_se_mtl_cnn_weights.pth", map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)

    with open(weights_dir / "fault_encoder.pkl", "rb") as f:
        fault_enc = pickle.load(f)
    with open(weights_dir / "maneuver_encoder.pkl", "rb") as f:
        maneuver_enc = pickle.load(f)
    with open(weights_dir / "scaler_mel.pkl", "rb") as f:
        scaler_mel = pickle.load(f)
    with open(weights_dir / "scaler_custom.pkl", "rb") as f:
        scaler_stft = pickle.load(f)

    return model, fault_enc, maneuver_enc, scaler_mel, scaler_stft


def predict(audio_path: str, device: str = "cpu") -> dict:
    """Классификация одного аудиофайла. Возвращает словарь с предсказаниями."""
    model, fault_enc, maneuver_enc, scaler_mel, scaler_stft = load_pipeline(device=device)

    mel, stft = extract_all(audio_path)
    # Применяем сохранённые скейлеры (обучены на train выборке)
    mel_flat = mel.reshape(-1)
    mel_scaled = scaler_mel.transform(mel_flat.reshape(1, -1)).reshape(mel.shape)
    stft_scaled = scaler_stft.transform(stft.reshape(1, -1))

    with torch.no_grad():
        mel_t = torch.tensor(mel_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        stft_t = torch.tensor(stft_scaled, dtype=torch.float32).to(device)
        fault_logits, maneuver_logits = model(mel_t, stft_t)
        fault_probs = torch.softmax(fault_logits, dim=1)[0].cpu().numpy()
        maneuver_probs = torch.softmax(maneuver_logits, dim=1)[0].cpu().numpy()

    fault_idx = int(fault_probs.argmax())
    maneuver_idx = int(maneuver_probs.argmax())

    return {
        "fault_class": fault_enc.classes_[fault_idx],
        "fault_confidence": float(fault_probs[fault_idx]),
        "fault_probs": dict(zip(fault_enc.classes_, fault_probs.tolist())),
        "maneuver_class": maneuver_enc.classes_[maneuver_idx],
        "maneuver_confidence": float(maneuver_probs[maneuver_idx]),
        "maneuver_probs": dict(zip(maneuver_enc.classes_, maneuver_probs.tolist())),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <path_to_audio.wav>")
        sys.exit(1)
    result = predict(sys.argv[1])
    print(f'Fault:    {result["fault_class"]} (confidence {result["fault_confidence"]:.4f})')
    print(f'Maneuver: {result["maneuver_class"]} (confidence {result["maneuver_confidence"]:.4f})')
    print("\nTop-3 fault probabilities:")
    sorted_fault = sorted(result["fault_probs"].items(), key=lambda kv: -kv[1])[:3]
    for cls, p in sorted_fault:
        print(f"  {cls}: {p:.4f}")
