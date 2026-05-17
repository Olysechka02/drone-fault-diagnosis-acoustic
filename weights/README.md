# Веса обученной модели

## Файлы

| Файл | Размер | Описание |
|---|---:|---|
| `fusion_resnet_se_mtl_cnn_weights.pth` | ~5.0 МБ | Веса FusionResNetSEMTLCNN (state_dict) |
| `fault_encoder.pkl` | ~0.5 КБ | sklearn LabelEncoder для 9 классов неисправностей |
| `maneuver_encoder.pkl` | ~0.5 КБ | sklearn LabelEncoder для 6 манёвров |
| `scaler_mel.pkl` | ~30 КБ | sklearn StandardScaler для мел-признаков (1280 dim) |
| `scaler_custom.pkl` | ~1.4 КБ | sklearn StandardScaler для STFT-статистик (32 dim) |

## Параметры

- Архитектура: `FusionResNetSEMTLCNN` (см. `src/model.py`);
- Число обучаемых параметров: 1 304 447;
- Формат: float32 (state_dict через `torch.save`);
- Тренировочная выборка: датасет Yi et al. (2023), 116 597 семплов (60 % от 194 329);
- Test fault accuracy: **97.02 %**, F1 = 0.9701;
- Test maneuver accuracy: 95.03 %, F1 = 0.9504.

## Использование

```python
from src.predict import predict

result = predict("data/samples/A_B_MF1_185_DuckPond_637_snr=13.18.wav")
print(result["fault_class"])         # 'MF1'
print(result["fault_confidence"])    # 0.97...
```

Или из командной строки:

```bash
python -m src.predict data/samples/A_B_MF1_185_DuckPond_637_snr=13.18.wav
```
