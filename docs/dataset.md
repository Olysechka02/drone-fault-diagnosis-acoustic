# Датасет

Модель `FusionResNetSEMTLCNN` обучена на датасете акустических записей квадрокоптеров из работы:

**Yi W., Choi J.-W., Lee J.-W.** *Sound-based drone fault classification using multitask learning.* — Proceedings of the 29th International Congress on Sound and Vibration (ICSV29). — Prague, 2023.
- arXiv: [2304.11708](https://arxiv.org/abs/2304.11708)
- Zenodo: [10.5281/zenodo.7779574](https://zenodo.org/records/7779574)

В этом репозитории сам датасет **не лежит** (он большой и уже опубликован отдельно с DOI). В `data/samples/` есть **3 файла-примера** для проверки инференса.

---

## Как скачать полный датасет

1. Перейди на страницу: https://zenodo.org/records/7779574
2. Скачай архив `Drone_Audio_Dataset.zip` (примерно несколько ГБ).
3. Распакуй в локальную папку, например в `data/raw/` внутри этого репо.
4. После распаковки структура должна быть примерно такой:

```
data/raw/
├── A/                          # Holy Stone HS720
│   ├── A_F_N_<index>_<bg>_<snr>.wav
│   ├── A_F_MF1_<index>_<bg>_<snr>.wav
│   └── …
├── B/                          # MJX Bugs 12 EIS
│   └── …
└── C/                          # ZLRC SG960 pro
    └── …
```

## Структура имени файла

```
<MODEL>_<MANEUVER>_<FAULT>_<INDEX>_<BACKGROUND>_<BG_INDEX>_snr=<SNR>.wav
```

- `MODEL` — `A` / `B` / `C` (тип квадрокоптера);
- `MANEUVER` — `F` (вперёд) / `B` (назад) / `L` (влево) / `R` (вправо) / `C` (по часовой) / `CC` (против часовой);
- `FAULT` — `N` (норма) / `MF1`…`MF4` (отказ мотора) / `PC1`…`PC4` (обрез пропеллера);
- `BACKGROUND` — одна из 5 локаций фонового шума (DuckPond, Hill, и т. д.);
- `SNR` — отношение сигнал/шум в децибелах (10–15 дБ).

## Параметры записи

- Частота дискретизации: **48 кГц**, передискретизирована до **16 кГц**;
- Длина сегмента: **0,5 с**;
- Микрофоны: **RØDE Wireless Go 2**, закреплённые на корпусе аппарата;
- Запись в безэховой камере, смешанная с фоновым шумом 5 точек кампуса при SNR 10–15 дБ;
- Всего: 54 000 × 2 файла на каждую из 3 моделей квадрокоптеров.

## Если хочешь обучить модель с нуля

После того как датасет скачан и распакован в `data/raw/`, потребуется:

1. **Извлечь признаки в parquet-файлы** двумя скриптами (находятся в `notebooks/final_model.ipynb` или взять из исходных `features1280.ipynb` и `features32_STFT+FFT.ipynb`):
   - мел-спектрограммы 20 × 64 = 1280 значений → `features_win20_hop12_mel64_feat1280/`;
   - кастомные STFT-статистики 32 значения → `features_win20_hop12_stft_feat32/`.
2. **Запустить обучение** — `notebooks/final_model.ipynb`, в `Config` указать пути к этим parquet-файлам.

**⚠️ Перед запуском обучения отредактируй пути в `Config` внутри ноутбука:**

```python
class Config:
    PARQUET_DIR_MEL = r"ВАШ_ПУТЬ\features_win20_hop12_mel64_feat1280\combined"
    PARQUET_DIR_CUSTOM = r"ВАШ_ПУТЬ\features_win20_hop12_stft_feat32\combined"
    BASE_RESULTS_DIR = r"ВАШ_ПУТЬ\для\результатов"
```

Обучение занимает ~82 минуты на одной NVIDIA GPU с CUDA 12.x.

## Если хочешь только инференс (без обучения)

Для инференса полный датасет **не нужен** — достаточно 3 примеров в `data/samples/` либо своих WAV-файлов длительностью 0,5 с при 16 кГц моно. Запуск:

```bash
python -m src.predict path/to/your/audio.wav
```

Результат — предсказанный класс неисправности и манёвра с уверенностью.
