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

result = predict("path/to/your/audio.wav")
print(result["fault_class"])         # 'MF1', 'PC2', 'N' и т. д.
print(result["fault_confidence"])    # уверенность 0.0–1.0
```

Или из командной строки:

```bash
python -m src.predict path/to/your/audio.wav
```

## Формат файлов

Все .pkl-файлы сохранены через **joblib** (стандарт для sklearn-объектов).
В коде используется `joblib.load(...)`, а не `pickle.load(...)` — это важно
для совместимости версий.

## Проверка совместимости

Модель в `src/model.py` точно соответствует архитектуре из
`notebooks/final_model.ipynb`. Веса `fusion_resnet_se_mtl_cnn_weights.pth`
содержат 119 ключей state_dict, все совпадают с моделью; общее число
обучаемых параметров — 1 304 447. Все 3 примера в `data/samples/`
классифицируются корректно (N, MF1, PC1) с confidence ≈ 1.0.
