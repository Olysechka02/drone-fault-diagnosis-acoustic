"""
Архитектура FusionResNetSEMTLCNN — двухпотоковая сеть для акустической
диагностики неисправностей малых БАС.

Архитектура:
    Мел-ветвь:   мел-спектрограмма 20×64 → Conv1D → 3×SEResidualBlock → AvgPool → FC(128→96)
    STFT-ветвь:  32 статистических признака → FC(32→32)
    Fusion:      конкатенация 96+32=128 → FC(128→64)
    Две головы:  Fault (9 классов) + Maneuver (6 классов)

Источники:
    He K. et al. Deep Residual Learning for Image Recognition // CVPR 2016.
    Hu J. et al. Squeeze-and-Excitation Networks // CVPR 2018.
    Caruana R. Multitask Learning // Machine Learning 28(1), 1997.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    """Squeeze-and-Excitation слой (Hu et al., 2018) для каналов 1D-свёртки.

    Args:
        channel: число входных каналов C.
        reduction: коэффициент сжатия r в bottleneck (по умолчанию 16).
    """

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        hidden = max(channel // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # Squeeze: (B, C, T) → (B, C)
        y = self.fc(y).view(b, c, 1)     # Excitation: (B, C) → (B, C, 1)
        return x * y                     # Scale: (B, C, T) ⊙ (B, C, 1)


class SEResidualBlock(nn.Module):
    """Остаточный блок ResNet с интегрированным SE-механизмом.

    Структура: Conv1D → BN → ReLU → Conv1D → BN → SELayer → [+ shortcut] → ReLU.
    Shortcut — тождественный или проекционный (1×1 Conv + BN) при изменении C/stride.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SELayer(out_ch, reduction=reduction)

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + identity, inplace=True)


class FusionResNetSEMTLCNN(nn.Module):
    """Двухпотоковая модель с SE-блоками и мультизадачным выходом.

    Args:
        n_fault: число классов неисправностей (по умолчанию 9: N + MF1-4 + PC1-4).
        n_maneuver: число классов манёвров (по умолчанию 6: F, B, L, R, C, CC).
        mel_frames: число временных кадров мел-спектрограммы (по умолчанию 20).
        mel_bins: число мел-полос (по умолчанию 64).
        stft_dim: размерность вектора STFT-статистик (по умолчанию 32).
        dropout: dropout в общем слое (по умолчанию 0.3).
    """

    def __init__(
        self,
        n_fault: int = 9,
        n_maneuver: int = 6,
        mel_frames: int = 20,
        mel_bins: int = 64,
        stft_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mel_frames = mel_frames
        self.mel_bins = mel_bins

        # ── Мел-ветвь ────────────────────────────────────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv1d(mel_bins, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            SEResidualBlock(32, 32),
            SEResidualBlock(32, 32),
        )
        self.stage2 = nn.Sequential(
            SEResidualBlock(32, 64, stride=2),
            SEResidualBlock(64, 64),
        )
        self.stage3 = nn.Sequential(
            SEResidualBlock(64, 128, stride=2),
            SEResidualBlock(128, 128),
        )
        self.mel_pool = nn.AdaptiveAvgPool1d(1)
        self.mel_fc = nn.Sequential(
            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── STFT-ветвь ───────────────────────────────────────────────────────
        self.stft_fc = nn.Sequential(
            nn.Linear(stft_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # ── Fusion + общий слой ──────────────────────────────────────────────
        self.shared_fc = nn.Sequential(
            nn.Linear(96 + 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Две головы (MTL) ─────────────────────────────────────────────────
        self.fault_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_fault),
        )
        self.maneuver_head = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, n_maneuver),
        )

    def forward(self, mel: torch.Tensor, stft: torch.Tensor):
        """
        Args:
            mel: тензор мел-спектрограмм (B, mel_frames, mel_bins) или (B, mel_frames*mel_bins).
            stft: тензор STFT-статистик (B, stft_dim).
        Returns:
            (fault_logits, maneuver_logits) — оба (B, n_classes).
        """
        # Если на вход подан плоский вектор 1280 — пересобираем в (B, 20, 64)
        if mel.dim() == 2 and mel.size(1) == self.mel_frames * self.mel_bins:
            mel = mel.view(-1, self.mel_frames, self.mel_bins)
        # Переставляем оси: (B, T, F) → (B, F, T) для Conv1D по времени
        mel = mel.transpose(1, 2)

        x = self.conv1(mel)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.mel_pool(x).squeeze(-1)
        mel_feat = self.mel_fc(x)

        stft_feat = self.stft_fc(stft)

        fused = torch.cat([mel_feat, stft_feat], dim=1)
        shared = self.shared_fc(fused)

        return self.fault_head(shared), self.maneuver_head(shared)


def count_parameters(model: nn.Module) -> int:
    """Подсчитать число обучаемых параметров."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Демонстрация
    model = FusionResNetSEMTLCNN()
    print(f"Параметров: {count_parameters(model):,}")  # ~1 304 447

    mel = torch.randn(4, 20, 64)
    stft = torch.randn(4, 32)
    fault, maneuver = model(mel, stft)
    print(f"Fault logits: {fault.shape}")      # (4, 9)
    print(f"Maneuver logits: {maneuver.shape}")  # (4, 6)
