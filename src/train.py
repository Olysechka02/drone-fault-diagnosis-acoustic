"""
Цикл обучения FusionResNetSEMTLCNN на датасете Yi et al. (2023).

Этот файл — рабочий каркас. Полный код обучения с обработкой данных, метриками
и сохранением весов находится в `notebooks/final_model.ipynb`. Здесь — основные
функции, которые можно переиспользовать.

Параметры обучения (из ВКР):
    Batch size: 128
    Optimizer: Adam (lr=1e-3, β1=0.9, β2=0.999)
    Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
    Loss: 0.8 * CE_fault + 0.2 * CE_maneuver
    Epochs: 50, early stopping patience=10
    Stratified split 6:2:2 по составному ключу model × fault × maneuver
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import FusionResNetSEMTLCNN


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    alpha: float = 0.8,
) -> dict:
    """Одна эпоха обучения.

    Каждый батч из loader возвращает (mel, stft, fault_label, maneuver_label).
    Возвращает средние losses и accuracies по эпохе.
    """
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, total_fault_loss, total_maneuver_loss = 0.0, 0.0, 0.0
    correct_fault, correct_maneuver, total_n = 0, 0, 0

    for mel, stft, y_fault, y_maneuver in loader:
        mel = mel.to(device)
        stft = stft.to(device)
        y_fault = y_fault.to(device)
        y_maneuver = y_maneuver.to(device)

        optimizer.zero_grad()
        logits_fault, logits_maneuver = model(mel, stft)

        loss_fault = ce(logits_fault, y_fault)
        loss_maneuver = ce(logits_maneuver, y_maneuver)
        loss = alpha * loss_fault + (1 - alpha) * loss_maneuver

        loss.backward()
        optimizer.step()

        bs = y_fault.size(0)
        total_loss += loss.item() * bs
        total_fault_loss += loss_fault.item() * bs
        total_maneuver_loss += loss_maneuver.item() * bs
        correct_fault += (logits_fault.argmax(1) == y_fault).sum().item()
        correct_maneuver += (logits_maneuver.argmax(1) == y_maneuver).sum().item()
        total_n += bs

    return {
        "loss": total_loss / total_n,
        "fault_loss": total_fault_loss / total_n,
        "maneuver_loss": total_maneuver_loss / total_n,
        "fault_acc": correct_fault / total_n,
        "maneuver_acc": correct_maneuver / total_n,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, alpha: float = 0.8) -> dict:
    """Оценка модели на отложенной выборке."""
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct_fault, correct_maneuver, total_n = 0, 0, 0

    for mel, stft, y_fault, y_maneuver in loader:
        mel = mel.to(device)
        stft = stft.to(device)
        y_fault = y_fault.to(device)
        y_maneuver = y_maneuver.to(device)

        logits_fault, logits_maneuver = model(mel, stft)
        loss = alpha * ce(logits_fault, y_fault) + (1 - alpha) * ce(logits_maneuver, y_maneuver)

        bs = y_fault.size(0)
        total_loss += loss.item() * bs
        correct_fault += (logits_fault.argmax(1) == y_fault).sum().item()
        correct_maneuver += (logits_maneuver.argmax(1) == y_maneuver).sum().item()
        total_n += bs

    return {
        "loss": total_loss / total_n,
        "fault_acc": correct_fault / total_n,
        "maneuver_acc": correct_maneuver / total_n,
    }


if __name__ == "__main__":
    print("Полный код обучения в notebooks/final_model.ipynb")
    print("Этот модуль предоставляет train_one_epoch() и evaluate() для переиспользования.")
